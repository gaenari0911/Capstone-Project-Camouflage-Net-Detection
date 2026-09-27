from __future__ import annotations

import gc
from pathlib import Path

import torch

from capstone_lab.config import sha256_file
from capstone_lab.errors import ArtifactError
from capstone_lab.training.checkpoints import load_training_checkpoint, save_training_checkpoint
from capstone_lab.training.dualhead import set_reproducible_seed, tensor_state_hash
from .contracts import approved_plan, code_records, digest, output_path
from .data import ReadOnlySegDataset, effective_args, make_loader, manifest_records, model_batch
from .io import atomic_json
from .models import create_model


def update(model, batch, optimizer, scaler):
    for retry in range(17):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model.loss(batch)[0].sum()
        if not torch.isfinite(loss):
            raise ArtifactError("non-finite resume test loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale:
            return float(loss.detach())
    raise ArtifactError("AMP failed to stabilize")


def verify_real_resume(root, output):
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    _, _, sources = approved_plan(root)
    binding = {"source_sha256": digest(sources), "code_sha256": digest(code_records(root)), "scope": "S8_REAL_CHECKPOINT_PREFLIGHT"}
    output.mkdir(parents=True, exist_ok=True)
    args = effective_args(root)
    dataset = ReadOnlySegDataset(root, manifest_records(root, "low187"), args, augment=True)
    set_reproducible_seed(920000)
    raw = next(iter(make_loader(dataset, seed=0, shuffle=True)))
    batch = model_batch(raw, "H2", "cuda:0")
    initial_dir = output / "continuous"
    initial_dir.mkdir()
    model, initial = create_model(root, initial_dir, 0, "H2")
    model.to("cuda:0").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=.00001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    first = update(model, batch, optimizer, scaler)
    scheduler.step()
    checkpoint = output / "step1.pt"
    checkpoint_hash = save_training_checkpoint(checkpoint, model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, epoch=0, global_step=1, contract=binding)
    second = update(model, batch, optimizer, scaler)
    scheduler.step()
    continuous_hash = tensor_state_hash(model.state_dict())
    continuous_state = {n: v.detach().cpu().clone() for n,v in model.state_dict().items()}
    del model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()
    resume_dir = output / "resumed"
    resume_dir.mkdir()
    model, resumed_initial = create_model(root, resume_dir, 0, "H2")
    model.to("cuda:0").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=.00001)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    restored = load_training_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                        scaler=scaler, expected_contract=binding)
    resumed_loss = update(model, batch, optimizer, scaler)
    scheduler.step()
    resumed_hash = tensor_state_hash(model.state_dict())
    differences = []
    for name, value in model.state_dict().items():
        actual = value.detach().cpu()
        expected = continuous_state[name]
        if not torch.equal(actual, expected):
            differences.append({"name": name, "max_abs": float((actual.float()-expected.float()).abs().max())})
    atomic_json(output / "comparison.json", {"initial_equal": initial == resumed_initial,
                "second_loss": second, "resumed_loss": resumed_loss, "differences": differences}, immutable=True)
    if resumed_hash != continuous_hash or initial != resumed_initial or resumed_loss != second:
        raise ArtifactError("real batch checkpoint continuation differs from uninterrupted update")
    mismatch_refused = False
    try:
        load_training_checkpoint(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler,
                                 scaler=scaler, expected_contract={**binding, "code_sha256": "0" * 64})
    except ArtifactError:
        mismatch_refused = True
    if not mismatch_refused:
        raise ArtifactError("changed checkpoint contract accepted")
    result = {"status": "VERIFIED", "scope": "real_batch_step_resume_not_formal_epoch_resume",
              "contract": binding, "checkpoint_sha256": checkpoint_hash,
              "first_loss": first, "continuous_second_loss": second, "resumed_second_loss": resumed_loss,
              "continuous_state_sha256": continuous_hash, "resumed_state_sha256": resumed_hash,
              "restored": restored, "changed_contract_refused": mismatch_refused,
              "limitations": ["epoch/DataLoader order resume not yet implemented", "not a full training run"]}
    atomic_json(output / "summary.json", result, immutable=True)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    print(verify_real_resume(root, output_path(root, args.output))["status"])


if __name__ == "__main__":
    main()
