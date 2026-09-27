"""Real-only epoch adapter. No Test paths, synthetic stand-ins, or implicit recipe tuning."""
from __future__ import annotations

import copy
import gc
import hashlib
import math
import os
import time
from pathlib import Path

import torch

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError
from capstone_lab.training.checkpoints import capture_rng_state, restore_rng_state
from capstone_lab.training.dualhead import tensor_state_hash
from .contracts import approved_plan, code_records, digest, inside, read_json
from .data import ReadOnlySegDataset, effective_args, make_loader, manifest_records, model_batch
from .io import RunLock, atomic_json, atomic_replace
from .models import create_model, validate_masks

REAL_EXPERIMENTS = {"M0": ("real748", "H2"), "L0": ("low187", "H2"),
                    "H0_R": ("real748", "H0"), "H1_R": ("real748", "H1")}
RECIPE = {"version": "s8_epoch_v1", "optimizer": "AdamW", "betas": [.9, .999],
          "eps": 1e-8, "lr0": .001, "weight_decay": .0005, "nbs": 16,
          "accumulation": 1, "decay": "non-normalization weights only; biases/norm no decay",
          "foreach": False, "fused": False, "gradient_clip_norm": 10.,
          "warmup_epochs": 3, "warmup": "all groups linear 0 to lr0, first update lr0/warmup_steps",
          "schedule": "per-update cosine after warmup, last update lr0*lrf", "lrf": .01,
          "workers": 2, "epoch_seed": "900000 + model_seed*10000 + zero_based_epoch",
          "persistent_workers": False, "drop_last": False, "ema": False,
          "best": "metrics/mAP50-95(M); strict greater; earliest tie", "early_stopping": False,
          "amp_overflow": "retry same batch after restoring model buffers and RNG; at most 17",
          "checkpoint": "two last + two best_mask slots, atomic commit pointer"}


def learning_rate(step, steps_per_epoch, epochs):
    warmup = 3 * steps_per_epoch
    if step < warmup:
        return .001 * (step + 1) / warmup
    remaining = max(1, epochs * steps_per_epoch - warmup - 1)
    progress = min(1., (step - warmup) / remaining)
    return .001 * (.01 + .99 * (1 + math.cos(math.pi * progress)) / 2)


def make_optimizer(model):
    groups = {"weight": [], "bias": [], "norm": []}
    names = {key: [] for key in groups}
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad:
                continue
            group = "bias" if name == "bias" else "norm" if isinstance(module, torch.nn.modules.batchnorm._NormBase) or "Norm" in type(module).__name__ else "weight"
            groups[group].append(parameter)
            names[group].append(module_name + "." + name)
    if len({id(p) for values in groups.values() for p in values}) != sum(len(v) for v in groups.values()):
        raise ArtifactError("optimizer parameter duplicated")
    optimizer = torch.optim.AdamW([{"params": values, "weight_decay": .0005 if key == "weight" else 0., "group_name": key}
                                   for key, values in groups.items()], lr=.001, betas=(.9, .999),
                                  eps=1e-8, foreach=False, fused=False)
    return optimizer, names


def epoch_update(model, batch, optimizer, scaler):
    # Overflow forwards update BN counters even when GradScaler skips the optimizer.
    # Restore those buffers AND the RNG before retrying the same batch.
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    rng = capture_rng_state()
    for retry in range(17):
        if retry:
            with torch.no_grad():
                for name, value in model.named_buffers():
                    value.copy_(buffers[name])
            restore_rng_state(rng)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.loss(batch)[0].sum()
        if not bool(torch.isfinite(loss)):
            raise ArtifactError("non-finite epoch loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if finite:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True, foreach=False)
        previous = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous:
            if not finite:
                raise ArtifactError("non-finite gradient accepted")
            return float(loss.detach()), retry
    raise ArtifactError("AMP overflow retry budget exhausted")


def save_tensor(path, payload):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    atomic_replace(temporary, path)
    return sha256_file(path)


def verify_pointer(directory, binding):
    pointer = read_json(directory / "current.json")
    if pointer.get("contract") != binding:
        raise ArtifactError("epoch checkpoint contract changed")
    for key in ("last", "best_mask"):
        record = pointer[key]
        if record["file"] not in {f"{key}_{i}.pt" for i in (0, 1)}:
            raise ArtifactError("invalid rolling checkpoint slot")
        if sha256_file(inside(directory, record["file"])) != record["sha256"]:
            raise ArtifactError("epoch checkpoint hash mismatch")
    return pointer


def commit_epoch(directory, model, optimizer, scaler, binding, history, previous, initial):
    epoch = history[-1]["epoch"]
    score = history[-1]["mask_map50_95"]
    slot = 1 - int(previous["last"]["file"][-4]) if previous else 0
    better = previous is None or score > previous["best_mask"]["score"]
    if better:
        best_slot = 1 - int(previous["best_mask"]["file"][-4]) if previous else 0
        name = f"best_mask_{best_slot}.pt"
        best_hash = save_tensor(directory / name, {"schema": 1, "contract": binding,
                                "model": model.state_dict(), "epoch": epoch, "score": score,
                                "initial": initial})
        best = {"file": name, "sha256": best_hash, "epoch": epoch, "score": score}
    else:
        best = previous["best_mask"]
    name = f"last_{slot}.pt"
    payload = {"schema": 1, "contract": binding, "model": model.state_dict(),
               "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
               "scheduler": {"policy": RECIPE["schedule"], "global_step": history[-1]["global_step"]},
               "rng": capture_rng_state(), "epoch": epoch, "history": history, "initial": initial,
               "best_mask": best}
    last_hash = save_tensor(directory / name, payload)
    pointer = {"schema": 1, "contract": binding, "epoch": epoch,
               "last": {"file": name, "sha256": last_hash}, "best_mask": best,
               "state_sha256": tensor_state_hash(model.state_dict())}
    atomic_json(directory / "current.json", pointer)
    return pointer


def training_binding(root, experiment, seed, epochs, smoke):
    import ultralytics
    _, _, inputs = approved_plan(root)
    library = Path(ultralytics.__file__).parent
    args = effective_args(root)
    args.workers, args.seed, args.epochs = 2, seed, epochs
    from .mixed import MIXTURES, binding as mixed_binding
    mixture = mixed_binding(root, experiment) if experiment in MIXTURES else None
    from .generative_data import EXPERIMENTS as GENERATIVE, binding as generative_binding
    if experiment in GENERATIVE:
        mixture = generative_binding(root, experiment)
    return {"experiment": experiment, "seed": seed, "epochs": epochs,
            "scope": "EPOCH_PREFLIGHT_NOT_FORMAL" if smoke else ("FORMAL_GENERATIVE_TRAINING" if experiment in GENERATIVE else "FORMAL_REAL_TRAINING"),
            "recipe": RECIPE, "effective_args": vars(args),
            "training_subset": mixture if mixture else ("low187" if smoke else REAL_EXPERIMENTS[experiment][0]),
            "inputs": inputs, "code": code_records(Path(__file__).resolve().parents[2]), "torch": str(torch.__version__),
            "library": {p.relative_to(library).as_posix(): sha256_file(p) for p in sorted(library.rglob("*.py"))}}


def train_epochs(root, directory, *, experiment, seed, epochs, smoke=False, stop_after=None,
                 should_pause=None, progress_path=None):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    from .mixed import MIXTURES, records as mixed_records
    from .generative_data import EXPERIMENTS as GENERATIVE, records as generative_records, require_release
    generative = experiment in GENERATIVE
    if generative and not smoke:
        require_release(root)
    mixed = experiment in MIXTURES
    if mixed and not smoke:
        gate = read_json(root / 'artifacts/s8_mixed_release/run_v2/summary.json')
        if gate.get('status') != 'VERIFIED_MIXED_EPOCH_RELEASE' or gate.get('code_sha256') != digest(code_records(Path(__file__).resolve().parents[2])):
            raise ArtifactError('Mixed formal release/code not verified')
    expected_epochs = MIXTURES[experiment][4] if mixed else 150
    if (experiment not in REAL_EXPERIMENTS and not mixed and not generative) or seed not in (0, 1, 2) or (epochs != expected_epochs and not smoke) or (smoke and epochs != 2):
        raise ArtifactError("outside registered real-only epoch scope")
    if mixed and (experiment.startswith('A') or experiment.startswith('H')) and seed != 0:
        raise ArtifactError('A/H seeds restricted to zero')
    subset, variant = (GENERATIVE[experiment][0], 'H2') if generative else ((MIXTURES[experiment][0], MIXTURES[experiment][3]) if mixed else REAL_EXPERIMENTS[experiment])
    if smoke:
        subset = "low187"  # 2 complete 187-image epochs; never counted as M0/H0_R formal training.
    directory.mkdir(parents=True, exist_ok=True)
    with RunLock(directory / "training.lock"):
        binding = training_binding(root, experiment, seed, epochs, smoke)
        atomic_json(directory / "contract.json", binding, immutable=True)
        args = effective_args(root)
        args.workers, args.seed, args.epochs = 2, seed, epochs
        model, initial = create_model(root, directory, seed, variant)
        model.args = args
        model.to("cuda:0").train()
        optimizer, groups = make_optimizer(model)
        atomic_json(directory / "optimizer_groups.json", groups, immutable=True)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        history, pointer, global_step = [], None, 0
        if (directory / "current.json").exists():
            pointer = verify_pointer(directory, binding)
            saved = torch.load(directory / pointer["last"]["file"], map_location="cpu", weights_only=False)
            if saved["contract"] != binding or saved["initial"] != initial or saved["best_mask"] != pointer["best_mask"]:
                raise ArtifactError("checkpoint payload identity mismatch")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            scaler.load_state_dict(saved["scaler"])
            restore_rng_state(saved["rng"])
            history = saved["history"]
            global_step = saved["scheduler"]["global_step"]
            del saved
        records = generative_records(root, experiment, smoke) if generative else (mixed_records(root, experiment) if mixed else manifest_records(root, subset))
        validation = manifest_records(root, "val100")
        start_epoch = len(history)
        for epoch in range(start_epoch, epochs):
            if should_pause and should_pause() and pointer:
                return {"status": "PAUSED", "epoch": len(history), "contract_sha256": digest(binding)}
            started = time.monotonic()
            # Reconstruct every epoch, so worker-local transform RNG and cached images
            # have the same start state after process restart and uninterrupted execution.
            rng = capture_rng_state()
            torch.manual_seed(900000 + seed * 10000 + epoch)
            train = ReadOnlySegDataset(root, records, args, augment=True)
            restore_rng_state(rng)
            loader = make_loader(train, seed=seed * 10000 + epoch, workers=2, shuffle=True)
            model.train()
            torch.cuda.reset_peak_memory_stats()
            losses, overflow_retries = [], 0
            stream = hashlib.sha256()
            for batch_index, raw in enumerate(loader):
                for key in ("img", "masks", "batch_idx", "cls", "bboxes", "boundaries", "distance_maps"):
                    if key in raw:
                        stream.update(raw[key].contiguous().numpy().tobytes())
                stream.update(digest(list(raw["im_file"])).encode())
                batch = model_batch(raw, variant, "cuda:0")
                lr = learning_rate(global_step, len(loader), epochs)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                loss, retries = epoch_update(model, batch, optimizer, scaler)
                global_step += 1
                losses.append(loss)
                overflow_retries += retries
                if progress_path:
                    atomic_json(progress_path, {"epoch": epoch + 1, "epochs": epochs,
                                "batch": batch_index + 1, "batches": len(loader), "global_step": global_step,
                                "loss": loss, "updated": time.time(), "scope": binding["scope"],
                                "eta_seconds_after_current_epoch": (epochs-epoch-1) * sum(h["wall_seconds"] for h in history[-5:]) / len(history[-5:]) if history else None})
                del batch, raw
            train_peak = torch.cuda.max_memory_reserved() // 2**20
            optimizer.zero_grad(set_to_none=True)
            del loader, train
            gc.collect()
            torch.cuda.empty_cache()
            val = ReadOnlySegDataset(root, validation, args, augment=False)
            metrics = validate_masks(model, make_loader(val, seed=seed, workers=2), directory / "validation", args)
            del val
            score = metrics["library_metrics"]["metrics/mAP50-95(M)"]
            if not math.isfinite(score) or metrics["images"] != 100:
                raise ArtifactError("invalid complete Val100 metric")
            history.append({"epoch": epoch + 1, "global_step": global_step, "loss": sum(losses) / len(losses),
                            "mask_map50_95": score, "metrics": metrics, "data_stream_sha256": stream.hexdigest(),
                            "train_peak_reserved_mib": train_peak, "amp_retries": overflow_retries,
                            "lr_last": lr, "wall_seconds": time.monotonic() - started})
            pointer = commit_epoch(directory, model, optimizer, scaler, binding, history, pointer, initial)
            atomic_json(directory / "history.json", history)
            if stop_after and epoch + 1 >= stop_after and epoch + 1 < epochs:
                return {"status": "PAUSED", "epoch": epoch + 1, "contract_sha256": digest(binding)}
        result = {"status": "SUCCEEDED", "scope": binding["scope"], "experiment": experiment,
                  "seed": seed, "epochs": epochs, "resumed_from_epoch": start_epoch,
                  "contract_sha256": digest(binding), "initial": initial, "history": history,
                  "state_sha256": pointer["state_sha256"], "best_mask": pointer["best_mask"],
                  "last": pointer["last"], "formal_training_completed": not smoke}
        # Completion is written once; restart after a committed final epoch can recreate it.
        if (directory / "result.json").exists():
            old = read_json(directory / "result.json")
            if old["state_sha256"] != result["state_sha256"] or old["contract_sha256"] != result["contract_sha256"]:
                raise ArtifactError("completed epoch output mismatch")
            return old
        atomic_json(directory / "result.json", result, immutable=True)
        return result
