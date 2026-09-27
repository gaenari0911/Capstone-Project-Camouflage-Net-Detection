from __future__ import annotations

import ast
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from capstone_lab.errors import ManifestError

CONDITIONS = ("A0", "A1", "A2", "A3", "A4", "A5")
COMPONENTS = ("tone", "texture", "edge", "placement")
ARTIFACT_PENALTY_WEIGHT = 0.16
MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "natural": {"tone": 0.35, "texture": 0.20, "edge": 0.30, "placement": 0.15},
    "semi": {"tone": 0.30, "texture": 0.25, "edge": 0.25, "placement": 0.20},
    "hard": {"tone": 0.20, "texture": 0.25, "edge": 0.20, "placement": 0.35},
    "minimal": {"tone": 0.15, "texture": 0.20, "edge": 0.15, "placement": 0.50},
}
DISABLED_COMPONENT = {
    "A1": None,
    "A2": "tone",
    "A3": "texture",
    "A4": "edge",
    "A5": "placement",
}


def derive_seed(
    campaign_seed: int,
    sample_id: str,
    stage: str,
    attempt: int,
    *,
    condition_id: str | None = None,
) -> tuple[str, int]:
    if isinstance(campaign_seed, bool) or not isinstance(campaign_seed, int):
        raise ManifestError("campaign_seed must be an integer")
    if not sample_id or not stage or attempt < 0:
        raise ManifestError("seed fields must be non-empty and attempt must be nonnegative")
    parts = [str(campaign_seed), sample_id, stage, str(attempt)]
    if condition_id is not None:
        if condition_id not in CONDITIONS:
            raise ManifestError(f"unsupported condition: {condition_id}")
        parts.append(condition_id)
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest, int(digest, 16)


@dataclass(frozen=True)
class FixtureConfig:
    campaign_seed: int = 7
    sample_count: int = 4
    candidates_per_sample: int = 4
    positions_per_candidate: int = 3
    background_ids: tuple[str, ...] = (
        "bg_forest_001",
        "bg_grass_001",
        "bg_snow_001",
    )
    foreground_ids: tuple[str, ...] = (
        "fg_001",
        "fg_002",
        "fg_003",
        "fg_004",
        "fg_005",
        "fg_006",
    )
    modes: tuple[str, ...] = ("natural", "semi")
    domains: tuple[str, ...] = ("forest_dense", "grass_field", "snow")
    difficulties: tuple[str, ...] = ("easy", "medium", "hard", "extreme")
    scale_buckets: tuple[str, ...] = (
        "00to05",
        "05to10",
        "10to15",
        "15to20",
        "20to25",
        "25to30",
        "30to40",
        "40to60",
    )

    def validate(self) -> None:
        if self.sample_count < 1 or self.sample_count > 100:
            raise ManifestError("fixture sample_count must be between 1 and 100")
        if self.candidates_per_sample < 1 or self.positions_per_candidate < 1:
            raise ManifestError("fixture candidate counts must be positive")
        if self.candidates_per_sample > len(self.foreground_ids):
            raise ManifestError("fixture needs enough unique foreground IDs")
        if not self.background_ids or not self.modes or not self.domains:
            raise ManifestError("fixture pools must not be empty")
        if any(mode not in MODE_WEIGHTS for mode in self.modes):
            raise ManifestError("fixture contains an unsupported legacy mode")


@dataclass(frozen=True)
class CandidatePlan:
    sample_id: str
    background_id: str
    mode: str
    domain: str
    difficulty: str
    scale_bucket: str
    scale_ratio: float
    foreground_candidate_ids: tuple[str, ...]
    position_candidate_ids: tuple[str, ...]
    candidate_seed: str
    rendering_seed: str
    attempt: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["foreground_candidate_ids"] = list(self.foreground_candidate_ids)
        value["position_candidate_ids"] = list(self.position_candidate_ids)
        return value


@dataclass(frozen=True)
class CandidateFeature:
    sample_id: str
    candidate_id: str
    foreground_id: str
    position_id: str
    tone: float
    texture: float
    edge: float
    placement: float
    artifact_penalty: float
    eligible: bool
    eligibility_reasons: tuple[str, ...]

    def raw_components(self) -> dict[str, float]:
        return {
            "tone": self.tone,
            "texture": self.texture,
            "edge": self.edge,
            "placement": self.placement,
        }


def build_candidate_plans(config: FixtureConfig) -> list[CandidatePlan]:
    config.validate()
    plans: list[CandidatePlan] = []
    for index in range(1, config.sample_count + 1):
        sample_id = f"sample_{index:06d}"
        candidate_digest, candidate_seed = derive_seed(
            config.campaign_seed, sample_id, "candidate_plan", 0
        )
        rendering_digest, _ = derive_seed(
            config.campaign_seed, sample_id, "rendering", 0
        )
        rng = random.Random(candidate_seed)
        foregrounds = tuple(
            rng.sample(list(config.foreground_ids), config.candidates_per_sample)
        )
        positions = tuple(
            f"pos_{position_index:02d}_{rng.randrange(0, 10000):04d}"
            for position_index in range(config.positions_per_candidate)
        )
        scale_bucket = rng.choice(config.scale_buckets)
        scale_low, scale_high = {
            "00to05": (0.008, 0.05),
            "05to10": (0.05, 0.10),
            "10to15": (0.10, 0.15),
            "15to20": (0.15, 0.20),
            "20to25": (0.20, 0.25),
            "25to30": (0.25, 0.30),
            "30to40": (0.30, 0.40),
            "40to60": (0.40, 0.60),
        }[scale_bucket]
        plans.append(
            CandidatePlan(
                sample_id=sample_id,
                background_id=rng.choice(config.background_ids),
                mode=rng.choice(config.modes),
                domain=rng.choice(config.domains),
                difficulty=rng.choice(config.difficulties),
                scale_bucket=scale_bucket,
                scale_ratio=round(rng.uniform(scale_low, scale_high), 12),
                foreground_candidate_ids=foregrounds,
                position_candidate_ids=positions,
                candidate_seed=candidate_digest,
                rendering_seed=rendering_digest,
                attempt=0,
            )
        )
    return plans


def read_legacy_contract(source_path: Path) -> dict[str, Any]:
    source = source_path.resolve(strict=True).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments: dict[str, list[tuple[int, Any]]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id not in {"ARTIFACT_PENALTY_WEIGHT", "MODE_CONFIG"}:
            continue
        assignments.setdefault(target.id, []).append(
            (node.lineno, ast.literal_eval(node.value))
        )
    artifact = assignments.get("ARTIFACT_PENALTY_WEIGHT", [])
    modes = assignments.get("MODE_CONFIG", [])
    if not artifact or not modes:
        raise ManifestError("legacy score constants could not be parsed")
    extracted_weights = {
        name: dict(values["weights"]) for name, values in modes[-1][1].items()
    }
    return {
        "artifact_assignments": artifact,
        "effective_artifact_penalty_weight": artifact[-1][1],
        "effective_artifact_line": artifact[-1][0],
        "mode_config_line": modes[-1][0],
        "mode_weights": extracted_weights,
    }


def verify_legacy_contract(source_path: Path) -> dict[str, Any]:
    extracted = read_legacy_contract(source_path)
    if extracted["effective_artifact_penalty_weight"] != ARTIFACT_PENALTY_WEIGHT:
        raise ManifestError("adapter artifact penalty differs from legacy source")
    if extracted["mode_weights"] != MODE_WEIGHTS:
        raise ManifestError("adapter mode weights differ from legacy source")
    return extracted
