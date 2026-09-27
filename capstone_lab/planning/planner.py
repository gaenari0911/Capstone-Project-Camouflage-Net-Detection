from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from capstone_lab.config import canonical_json, sha256_file
from capstone_lab.errors import ArtifactError, ConfigError


TARGETS = ("ablation", "main", "low", "head_ablation", "core", "all")
SERIES = ("A", "M", "L", "H")


@dataclass(frozen=True)
class PlanRow:
    job_id: str
    experiment_id: str
    series: str
    seed: int
    node_kind: str
    model_identity: str
    head_identity: str
    loss_identity: str
    dataset_manifest_identity: str
    synthetic_dependency: str
    epochs: int
    resource_class: str
    development_scope: str
    approval_gates: tuple[str, ...]
    planned_artifact_uri: str
    reuse_target_job_id: str
    comparison_contract_sha256: str
    required_reuse_hashes: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["approval_gates"] = list(self.approval_gates)
        return value


@dataclass(frozen=True)
class S7Config:
    project_root: Path
    source_path: Path
    artifact_root: Path
    raw: dict[str, Any]
    input_paths: dict[str, Path]
    input_records: dict[str, dict[str, Any]]
    config_sha256: str
    input_sha256: str
    code_sha256: str


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def load_s7_config(path: Path, project_root: Path) -> S7Config:
    root = project_root.resolve(strict=True)
    source = path.resolve(strict=True)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ConfigError("S7 schema_version must equal 1")
    if raw.get("status") not in {"DRAFT", "BLOCKED_APPROVAL"}:
        raise ConfigError("S7 planning config must remain DRAFT or BLOCKED_APPROVAL")
    if raw.get("scope") != "PLANNING_ONLY_NO_EXECUTION":
        raise ConfigError("S7 may only create a planning dry-run")
    artifact = (root / raw["artifact_root"]).resolve(strict=False)
    allowed = (root / "artifacts" / "s7_plan").resolve(strict=False)
    if artifact == allowed or not _inside(artifact, allowed):
        raise ConfigError("S7 output must be a versioned directory below artifacts/s7_plan")
    protected = [root / "Dataset", root / "artifacts" / "s6_profile", root / "experiments"]
    if any(_inside(artifact, item.resolve(strict=False)) or _inside(item.resolve(strict=False), artifact) for item in protected):
        raise ConfigError("S7 output overlaps a protected source/artifact")
    inputs: dict[str, Path] = {}
    input_records: dict[str, dict[str, Any]] = {}
    for name, relative in raw["inputs"].items():
        candidate = (root / relative).resolve(strict=True)
        if not candidate.is_file() or _inside(candidate, artifact):
            raise ConfigError(f"invalid S7 input: {name}")
        digest = sha256_file(candidate)
        expected = raw["expected_input_hashes"].get(name)
        if digest != expected:
            raise ArtifactError(f"S7 input hash changed for {name}: {digest}")
        inputs[name] = candidate
        input_records[name] = {"path": str(candidate), "bytes": candidate.stat().st_size, "sha256": digest}
    planning = raw["planning"]
    if planning["seeds"] != [0, 1, 2] or planning["ablation_epochs"] != 40 or planning["full_epochs"] != 150:
        raise ConfigError("S7 counts require seeds 0/1/2, A=40 epochs, M/L/H=150 epochs")
    if planning["ablation_images_per_condition"] != 3000 or planning["generative_pool_images"] != 6000:
        raise ConfigError("S7 generation counts differ from the research plan")
    if planning["model_scale"] != "n" or planning["num_classes"] != 1:
        raise ConfigError("S7 learner must remain n-scale/nc=1")
    resources = raw["resource_candidates"]
    if resources["batch"] != 16 or resources["gpu_concurrency"] != 2 or resources["formal_config_frozen"] is not False:
        raise ConfigError("S6 resource findings must remain unfrozen candidates")
    if resources["s6_summary_sha256"] != input_records["s6_summary"]["sha256"]:
        raise ArtifactError("resource candidate does not reference the exact S6 summary")
    approvals = raw["approvals"]
    if set(approvals) != {"campaign_execution", "generative_pilot", "final_test"}:
        raise ConfigError("S7 approval gate set is incomplete")
    if any(value == "APPROVED" for value in approvals.values()):
        raise ConfigError("S7 may not grant execution, generative, or Test approval")
    code_files = sorted((root / "capstone_lab" / "planning").glob("*.py"))
    code_sha256 = _digest([
        {"path": item.relative_to(root).as_posix(), "sha256": sha256_file(item)}
        for item in code_files
    ])
    return S7Config(
        project_root=root,
        source_path=source,
        artifact_root=artifact,
        raw=raw,
        input_paths=inputs,
        input_records=input_records,
        config_sha256=_digest(raw),
        input_sha256=_digest(input_records),
        code_sha256=code_sha256,
    )


def _training_contract(config: S7Config, *, seed: int, dataset: str, epochs: int, head: str, loss: str) -> str:
    plan = config.raw["planning"]
    return _digest({
        "seed": seed,
        "dataset_manifest_identity": dataset,
        "epochs": epochs,
        "model": "custom_DualHeadSegment" if head != "H0_segment" else "custom_Segment_same_backbone_neck",
        "scale": plan["model_scale"],
        "num_classes": plan["num_classes"],
        "head": head,
        "loss": loss,
        "pretrained_sha256": plan["pretrained_sha256"],
        "initialization_policy": plan["initialization_policy"],
        "training": plan["training_contract"],
        "batch_candidate": config.raw["resource_candidates"]["batch"],
        "evaluation": plan["evaluation_contract"],
    })


def _row(
    config: S7Config,
    *,
    experiment: str,
    series: str,
    seed: int,
    dataset: str,
    synthetic: str,
    epochs: int,
    head: str = "H2_dual_distance_bce",
    loss: str = "distance_weighted_boundary_bce",
    kind: str = "TRAIN",
    reuse: str = "",
    contract_override: str | None = None,
) -> PlanRow:
    job_id = f"{experiment}_seed{seed}"
    contract = contract_override or _training_contract(config, seed=seed, dataset=dataset, epochs=epochs, head=head, loss=loss)
    artifact = f"formal://{reuse or job_id}" if kind == "REFERENCE" else f"formal://{job_id}"
    gates = ["campaign_execution"]
    if synthetic in {"generative_random_3000", "task_aware_3000"}:
        gates.append("generative_pilot")
    return PlanRow(
        job_id=job_id,
        experiment_id=experiment,
        series=series,
        seed=seed,
        node_kind=kind,
        model_identity="custom_DualHeadSegment_n_nc1" if head != "H0_segment" else "custom_Segment_n_nc1_same_backbone_neck",
        head_identity=head,
        loss_identity=loss,
        dataset_manifest_identity=dataset,
        synthetic_dependency=synthetic,
        epochs=0 if kind == "REFERENCE" else epochs,
        resource_class="REFERENCE" if kind == "REFERENCE" else "GPU_TRAIN_BATCH16_CANDIDATE",
        development_scope="validation_only_no_Test",
        approval_gates=tuple(gates),
        planned_artifact_uri=artifact,
        reuse_target_job_id=reuse,
        comparison_contract_sha256=contract,
        required_reuse_hashes="architecture,dataset,initialization,training,evaluation" if kind == "REFERENCE" else "",
    )


def build_registry(config: S7Config) -> list[PlanRow]:
    seeds = config.raw["planning"]["seeds"]
    a_epochs = config.raw["planning"]["ablation_epochs"]
    full_epochs = config.raw["planning"]["full_epochs"]
    real = f"sha256:{config.input_records['real_split_manifest']['sha256']}#split=train#count=748"
    low = f"sha256:{config.input_records['low187_manifest']['sha256']}#split=train#count=187"
    a_synth = {
        "A0": "random_copy_paste_3000",
        "A1": "A1_full_heuristic_3000",
        "A2": "A2_minus_tone_3000",
        "A3": "A3_minus_texture_3000",
        "A4": "A4_minus_edge_3000",
        "A5": "A5_minus_placement_3000",
    }
    rows: list[PlanRow] = []
    for experiment, synthetic in a_synth.items():
        dataset = f"{real}+planned://synthetic/{experiment}"
        for seed in seeds:
            rows.append(_row(config, experiment=experiment, series="A", seed=seed, dataset=dataset, synthetic=synthetic, epochs=a_epochs))
    strategies = {
        "M0": (real, "none"),
        "M1": (f"{real}+planned://synthetic/A1", "A1_full_heuristic_3000"),
        "M2": (f"{real}+planned://selection/generative_random", "generative_random_3000"),
        "M3": (f"{real}+planned://selection/task_aware", "task_aware_3000"),
        "L0": (low, "none"),
        "L1": (f"{low}+planned://synthetic/A1", "A1_full_heuristic_3000"),
        "L2": (f"{low}+planned://selection/generative_random", "generative_random_3000"),
        "L3": (f"{low}+planned://selection/task_aware", "task_aware_3000"),
    }
    for experiment, (dataset, synthetic) in strategies.items():
        for seed in seeds:
            rows.append(_row(config, experiment=experiment, series=experiment[0], seed=seed, dataset=dataset, synthetic=synthetic, epochs=full_epochs))
    by_job = {row.job_id: row for row in rows}
    mismatch = config.raw["planning"]["h2_reuse_fixture"] == "FORCED_MISMATCH"
    for head, loss in (("H0_segment", "no_boundary_loss"), ("H1_dual_bce", "plain_boundary_bce"), ("H2_dual_distance_bce", "distance_weighted_boundary_bce")):
        for suffix, source_exp in (("R", "M0"), ("S", "M1")):
            for seed in seeds:
                experiment = f"{head.split('_')[0]}_{suffix}"
                source = by_job[f"{source_exp}_seed{seed}"]
                dataset = source.dataset_manifest_identity
                target_contract = _training_contract(config, seed=seed, dataset=dataset, epochs=full_epochs, head=head, loss=loss)
                if head.startswith("H2") and mismatch:
                    target_contract = _digest({"base": target_contract, "fixture": "forced_mismatch"})
                if head.startswith("H2") and target_contract == source.comparison_contract_sha256:
                    rows.append(_row(config, experiment=experiment, series="H", seed=seed, dataset=dataset, synthetic=source.synthetic_dependency, epochs=full_epochs, head=head, loss=loss, kind="REFERENCE", reuse=source.job_id, contract_override=target_contract))
                else:
                    rows.append(_row(config, experiment=experiment, series="H", seed=seed, dataset=dataset, synthetic=source.synthetic_dependency, epochs=full_epochs, head=head, loss=loss, contract_override=target_contract))
    rows.sort(key=lambda row: (SERIES.index(row.series), row.experiment_id, row.seed, row.node_kind))
    validate_registry(rows)
    return rows


def validate_registry(rows: list[PlanRow]) -> None:
    ids = [row.job_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ConfigError("duplicate registry job ID")
    training_artifacts = [row.planned_artifact_uri for row in rows if row.node_kind == "TRAIN"]
    if len(training_artifacts) != len(set(training_artifacts)):
        raise ConfigError("duplicate training artifact URI")
    if any(row.development_scope != "validation_only_no_Test" for row in rows):
        raise ConfigError("Test leakage in a planning registry row")
    if any("#split=test" in row.dataset_manifest_identity.casefold() for row in rows):
        raise ConfigError("Test dataset may not enter training planning")
    by_id = {row.job_id: row for row in rows}
    for row in rows:
        if row.node_kind == "REFERENCE":
            source = by_id.get(row.reuse_target_job_id)
            if source is None or source.node_kind != "TRAIN":
                raise ConfigError("reference target is missing or is not a training row")
            if row.comparison_contract_sha256 != source.comparison_contract_sha256:
                raise ArtifactError("H2 reuse hash mismatch")


def _node(node_id: str, kind: str, dependencies: Iterable[str] = (), *, images: int = 0, gate: str = "") -> dict[str, Any]:
    return {"node_id": node_id, "kind": kind, "dependencies": sorted(set(dependencies)), "planned_unique_images": images, "approval_gate": gate}


def build_dag(rows: list[PlanRow]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {
        "gate_campaign_execution": _node("gate_campaign_execution", "APPROVAL", gate="campaign_execution"),
        "data_real748": _node("data_real748", "FROZEN_MANIFEST"),
        "data_low187": _node("data_low187", "FROZEN_MANIFEST"),
        "gate_generative_pilot": _node("gate_generative_pilot", "APPROVAL", gate="generative_pilot"),
    }
    for experiment in ("A0", "A1", "A2", "A3", "A4", "A5"):
        generate = f"generate_{experiment}_3000"
        validate = f"validate_{experiment}_3000"
        nodes[generate] = _node(generate, "CPU_SYNTHESIS", ("gate_campaign_execution",), images=3000)
        nodes[validate] = _node(validate, "SYNTHESIS_VALIDATION", (generate,))
    nodes["generate_generative_pool_6000"] = _node("generate_generative_pool_6000", "GENERATIVE_POOL", ("gate_campaign_execution", "gate_generative_pilot"), images=6000)
    nodes["select_generative_random_3000"] = _node("select_generative_random_3000", "SELECTION_MANIFEST", ("generate_generative_pool_6000",))
    nodes["judge_freeze_L0_seed0"] = _node("judge_freeze_L0_seed0", "JUDGE_FREEZE", ("L0_seed0",))
    nodes["select_task_aware_3000"] = _node("select_task_aware_3000", "SELECTION_MANIFEST", ("generate_generative_pool_6000", "judge_freeze_L0_seed0"))
    for row in rows:
        dependencies = ["gate_campaign_execution"]
        if row.series == "A":
            dependencies += ["data_real748", f"validate_{row.experiment_id}_3000"]
        elif row.experiment_id in {"M0", "L0"}:
            dependencies.append("data_real748" if row.series == "M" else "data_low187")
        elif row.experiment_id in {"M1", "L1"}:
            dependencies += ["data_real748" if row.series == "M" else "data_low187", "validate_A1_3000"]
        elif row.experiment_id in {"M2", "L2"}:
            dependencies += ["data_real748" if row.series == "M" else "data_low187", "select_generative_random_3000"]
        elif row.experiment_id in {"M3", "L3"}:
            dependencies += ["data_real748" if row.series == "M" else "data_low187", "select_task_aware_3000"]
        elif row.series == "H":
            if row.node_kind == "REFERENCE":
                dependencies = [row.reuse_target_job_id]
            else:
                dependencies += ["data_real748"]
                if row.experiment_id.endswith("_S"):
                    dependencies.append("validate_A1_3000")
        nodes[row.job_id] = _node(row.job_id, row.node_kind, dependencies)
    series_nodes = {series: [row.job_id for row in rows if row.series == series] for series in SERIES}
    for series in SERIES:
        nodes[f"aggregate_{series}"] = _node(f"aggregate_{series}", "VALIDATION_AGGREGATE", series_nodes[series])
    nodes["aggregate_core"] = _node("aggregate_core", "VALIDATION_AGGREGATE", ("aggregate_A", "aggregate_M", "aggregate_L"))
    nodes["aggregate_all"] = _node("aggregate_all", "VALIDATION_AGGREGATE", ("aggregate_core", "aggregate_H"))
    nodes["final_freeze_review"] = _node("final_freeze_review", "MANUAL_FREEZE", ("aggregate_all",))
    nodes["gate_final_Test"] = _node("gate_final_Test", "APPROVAL", ("final_freeze_review",), gate="final_test")
    validate_dag(nodes)
    return nodes


def validate_dag(nodes: dict[str, dict[str, Any]]) -> list[str]:
    for node_id, node in nodes.items():
        missing = set(node["dependencies"]) - set(nodes)
        if missing or node_id in node["dependencies"]:
            raise ConfigError(f"DAG node {node_id} has invalid dependencies: {sorted(missing)}")
    indegree = {key: 0 for key in nodes}
    children = {key: [] for key in nodes}
    for key, node in nodes.items():
        for dependency in node["dependencies"]:
            indegree[key] += 1
            children[dependency].append(key)
    ready = sorted(key for key, value in indegree.items() if value == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for child in sorted(children[current]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(nodes):
        raise ConfigError("DAG contains a cycle")
    return order


def target_closure(nodes: dict[str, dict[str, Any]], target: str, approvals: dict[str, str], requested_approvals: Iterable[str] = ()) -> dict[str, Any]:
    endpoints = {
        "ablation": "aggregate_A", "main": "aggregate_M", "low": "aggregate_L",
        "head_ablation": "aggregate_H", "core": "aggregate_core", "all": "final_freeze_review",
    }
    if target not in endpoints:
        raise ConfigError(f"unsupported target; Test cannot be selected through S7: {target}")
    requested = set(requested_approvals)
    unauthorized = {gate for gate in requested if approvals.get(gate) != "APPROVED"}
    if unauthorized:
        raise ConfigError(f"target cannot expand approval scope: {sorted(unauthorized)}")
    closure: set[str] = set()
    stack = [endpoints[target]]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        stack.extend(nodes[current]["dependencies"])
    order = [item for item in validate_dag(nodes) if item in closure]
    required_gates = sorted({nodes[item]["approval_gate"] for item in order if nodes[item]["approval_gate"]})
    blocked = [gate for gate in required_gates if approvals.get(gate) != "APPROVED"]
    if "gate_final_Test" in closure:
        raise ConfigError("S7 target closure leaked the final Test gate")
    return {"target": target, "endpoint": endpoints[target], "nodes": order, "required_approval_gates": required_gates, "blocked_approval_gates": blocked, "runnable": not blocked}


def failure_isolation(nodes: dict[str, dict[str, Any]], failed_node: str) -> dict[str, list[str]]:
    if failed_node not in nodes:
        raise ConfigError("unknown failed DAG node")
    blocked: set[str] = set()
    changed = True
    while changed:
        changed = False
        for node_id, node in nodes.items():
            if node_id == failed_node or node_id in blocked:
                continue
            if set(node["dependencies"]) & ({failed_node} | blocked):
                blocked.add(node_id)
                changed = True
    independent = sorted(set(nodes) - blocked - {failed_node})
    return {"blocked_descendants": sorted(blocked), "independent": independent}


def registry_csv(rows: list[PlanRow]) -> str:
    output = io.StringIO(newline="")
    fieldnames = list(rows[0].to_dict())
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        value = row.to_dict()
        value["approval_gates"] = "|".join(value["approval_gates"])
        writer.writerow(value)
    return output.getvalue()


def _totals(config: S7Config, rows: list[PlanRow], nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    training = [row for row in rows if row.node_kind == "TRAIN"]
    references = [row for row in rows if row.node_kind == "REFERENCE"]
    by_series = {
        series: {
            "logical_rows": sum(row.series == series for row in rows),
            "training_jobs": sum(row.series == series and row.node_kind == "TRAIN" for row in rows),
            "reference_rows": sum(row.series == series and row.node_kind == "REFERENCE" for row in rows),
            "epoch_job_units": sum(row.epochs for row in rows if row.series == series and row.node_kind == "TRAIN"),
        }
        for series in SERIES
    }
    unique_images = sum(node["planned_unique_images"] for node in nodes.values())
    storage = config.raw["storage_estimate"]
    synthesis_bytes = unique_images * int(storage["bytes_per_synthetic_bundle"])
    checkpoint_bytes = len(training) * int(storage["checkpoints_per_training_job"]) * int(storage["bytes_per_checkpoint"])
    generated_files = unique_images * int(storage["files_per_synthetic_bundle"])
    training_files = len(training) * int(storage["training_files_per_job"])
    subtotal = synthesis_bytes + checkpoint_bytes
    total_bytes = int(subtotal * (1 + float(storage["logs_and_margin_fraction"])))
    seed0 = [row for row in training if row.seed == 0]
    return {
        "registry_rows": len(rows),
        "core_training_jobs": sum(row.series in {"A", "M", "L"} and row.node_kind == "TRAIN" for row in rows),
        "H_logical_evaluation_cells": sum(row.series == "H" for row in rows),
        "H_additional_training_jobs": sum(row.series == "H" and row.node_kind == "TRAIN" for row in rows),
        "H_reference_rows": sum(row.series == "H" and row.node_kind == "REFERENCE" for row in rows),
        "unique_training_jobs": len(training),
        "unique_evaluation_cells_after_reuse": len(training),
        "epoch_job_units": sum(row.epochs for row in training),
        "seed0_training_jobs": len(seed0),
        "seed0_epoch_job_units": sum(row.epochs for row in seed0),
        "planned_unique_generated_images": unique_images,
        "planned_generated_bundle_files": generated_files,
        "planned_training_artifact_files": training_files,
        "planned_total_artifact_files": generated_files + training_files + 6,
        "estimated_synthesis_bytes": synthesis_bytes,
        "estimated_checkpoint_bytes": checkpoint_bytes,
        "estimated_total_with_margin_bytes": total_bytes,
        "estimated_total_with_margin_gib": total_bytes / 2**30,
        "by_series": by_series,
        "estimate_not_allocation": True,
    }


def build_plan(config: S7Config) -> dict[str, Any]:
    rows = build_registry(config)
    nodes = build_dag(rows)
    closures = {
        target: target_closure(nodes, target, config.raw["approvals"])
        for target in TARGETS
    }
    training_by_id = {row.job_id: row for row in rows if row.node_kind == "TRAIN"}
    for closure in closures.values():
        selected = [training_by_id[item] for item in closure["nodes"] if item in training_by_id]
        images = sum(nodes[item]["planned_unique_images"] for item in closure["nodes"])
        closure["training_jobs"] = len(selected)
        closure["epoch_job_units"] = sum(row.epochs for row in selected)
        closure["planned_unique_generated_images"] = images
        storage = config.raw["storage_estimate"]
        synthesis_bytes = images * int(storage["bytes_per_synthetic_bundle"])
        checkpoint_bytes = len(selected) * int(storage["checkpoints_per_training_job"]) * int(storage["bytes_per_checkpoint"])
        closure["estimated_artifact_files"] = (
            images * int(storage["files_per_synthetic_bundle"])
            + len(selected) * int(storage["training_files_per_job"])
        )
        closure["estimated_disk_bytes_with_margin"] = int(
            (synthesis_bytes + checkpoint_bytes)
            * (1 + float(storage["logs_and_margin_fraction"]))
        )
        closure["estimated_disk_gib_with_margin"] = closure["estimated_disk_bytes_with_margin"] / 2**30
    totals = _totals(config, rows, nodes)
    for series, target in (("A", "ablation"), ("M", "main"), ("L", "low"), ("H", "head_ablation")):
        totals["by_series"][series]["dependency_unique_images"] = closures[target]["planned_unique_generated_images"]
        totals["by_series"][series]["estimated_disk_gib_with_margin"] = closures[target]["estimated_disk_gib_with_margin"]
    matching = config.raw["planning"]["h2_reuse_fixture"] == "MATCHING"
    expected = (54, 6120, 6) if matching else (60, 7020, 0)
    observed = (totals["unique_training_jobs"], totals["epoch_job_units"], totals["H_reference_rows"])
    if observed != expected:
        raise ArtifactError(f"planned cost disagrees with research contract: {observed} != {expected}")
    checks = {
        "core_42_jobs_4320_units": totals["core_training_jobs"] == 42 and sum(totals["by_series"][x]["epoch_job_units"] for x in ("A", "M", "L")) == 4320,
        "H_18_cells_12_training_6_refs": totals["H_logical_evaluation_cells"] == 18 and totals["H_additional_training_jobs"] == (12 if matching else 18) and totals["H_reference_rows"] == (6 if matching else 0),
        "full_54_jobs_6120_units": (totals["unique_training_jobs"], totals["epoch_job_units"]) == expected[:2],
        "seed0_18_jobs_2040_units": totals["seed0_training_jobs"] == (18 if matching else 20) and totals["seed0_epoch_job_units"] == (2040 if matching else 2340),
        "unique_images_24000": totals["planned_unique_generated_images"] == 24000,
        "all_targets_blocked_approval": all(not item["runnable"] for item in closures.values()),
        "no_target_contains_Test_gate": all("gate_final_Test" not in item["nodes"] for item in closures.values()),
        "formal_resource_candidate_unfrozen": config.raw["resource_candidates"]["formal_config_frozen"] is False,
    }
    return {
        "schema_version": 1,
        "status": "VERIFIED" if all(checks.values()) else "PARTIAL_FAILURE",
        "scope": "S7 deterministic planning dry-run; no generation, training, GPU, inference, or Test",
        "runnable": False,
        "config_status": config.raw["status"],
        "contract": {"config_sha256": config.config_sha256, "input_sha256": config.input_sha256, "code_sha256": config.code_sha256},
        "input_records": config.input_records,
        "resource_candidates": config.raw["resource_candidates"],
        "approvals": config.raw["approvals"],
        "registry": [row.to_dict() for row in rows],
        "dag": {"nodes": [nodes[key] for key in sorted(nodes)], "topological_order": validate_dag(nodes)},
        "target_closures": closures,
        "totals": totals,
        "checks": checks,
    }
