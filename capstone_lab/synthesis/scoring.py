from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from capstone_lab.errors import ManifestError

from .contract import (
    ARTIFACT_PENALTY_WEIGHT,
    COMPONENTS,
    CONDITIONS,
    DISABLED_COMPONENT,
    MODE_WEIGHTS,
    CandidateFeature,
    CandidatePlan,
    derive_seed,
)

DOMAIN_PENALTY_WEIGHT = 0.08
BACKGROUND_PENALTY_WEIGHT = 0.18
RECENT_PENALTY_WEIGHT = 0.10
QUOTA_BONUS_WEIGHT = 0.06
COARSE_DOMAIN_PENALTY_WEIGHT = 0.30


@dataclass
class UsageState:
    foreground_ids: tuple[str, ...]
    max_usage: int = 5
    recent_size: int = 4
    used_total: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    used_by_domain: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    used_by_background: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    recent: deque[str] = field(default_factory=deque)

    def guidance(
        self, foreground_id: str, domain: str, background_id: str
    ) -> dict[str, float]:
        used = self.used_total.get(foreground_id, 0)
        remaining = max(0, self.max_usage - used)
        allowed_domain_count = 1 if domain == "snow" else 5
        return {
            "remaining_quota": float(remaining),
            "quota_bonus": remaining / float(max(1, self.max_usage)),
            "domain_repeat_penalty": self.used_by_domain.get(foreground_id, {}).get(
                domain, 0
            )
            / float(allowed_domain_count),
            "background_repeat_penalty": float(
                self.used_by_background.get(foreground_id, {}).get(background_id, 0)
            ),
            "recent_history_penalty": 1.0 if foreground_id in self.recent else 0.0,
            "coarse_domain_penalty": 0.0,
        }

    def commit(self, foreground_id: str, domain: str, background_id: str) -> None:
        self.used_total[foreground_id] += 1
        self.used_by_domain[foreground_id][domain] += 1
        self.used_by_background[foreground_id][background_id] += 1
        self.recent.append(foreground_id)
        while len(self.recent) > self.recent_size:
            self.recent.popleft()

    def snapshot(self) -> dict[str, Any]:
        return {
            "used_total": dict(sorted(self.used_total.items())),
            "used_by_domain": {
                key: dict(sorted(value.items()))
                for key, value in sorted(self.used_by_domain.items())
            },
            "used_by_background": {
                key: dict(sorted(value.items()))
                for key, value in sorted(self.used_by_background.items())
            },
            "recent": list(self.recent),
        }


def usage_adjustment(guidance: dict[str, float]) -> float:
    return (
        QUOTA_BONUS_WEIGHT * guidance["quota_bonus"]
        - DOMAIN_PENALTY_WEIGHT * guidance["domain_repeat_penalty"]
        - BACKGROUND_PENALTY_WEIGHT * guidance["background_repeat_penalty"]
        - RECENT_PENALTY_WEIGHT * guidance["recent_history_penalty"]
        - COARSE_DOMAIN_PENALTY_WEIGHT * guidance["coarse_domain_penalty"]
    )


def legacy_reference_final_score(
    feature: CandidateFeature, mode: str, guidance: dict[str, float]
) -> float:
    weights = MODE_WEIGHTS[mode]
    return (
        weights["tone"] * feature.tone
        + weights["texture"] * feature.texture
        + weights["edge"] * feature.edge
        + weights["placement"] * feature.placement
        - 0.16 * feature.artifact_penalty
        + 0.06 * guidance["quota_bonus"]
        - 0.08 * guidance["domain_repeat_penalty"]
        - 0.18 * guidance["background_repeat_penalty"]
        - 0.10 * guidance["recent_history_penalty"]
        - 0.30 * guidance["coarse_domain_penalty"]
    )


def score_candidate(
    condition_id: str,
    feature: CandidateFeature,
    mode: str,
    guidance: dict[str, float],
    *,
    sample_seed: str,
    attempt: int,
) -> dict[str, Any]:
    if condition_id not in DISABLED_COMPONENT:
        raise ManifestError(f"{condition_id} does not use score ranking")
    if mode not in MODE_WEIGHTS:
        raise ManifestError(f"unsupported mode: {mode}")
    disabled = DISABLED_COMPONENT[condition_id]
    configured = MODE_WEIGHTS[mode]
    enabled = {component: component != disabled for component in COMPONENTS}
    effective_weights = {
        component: configured[component] if enabled[component] else 0.0
        for component in COMPONENTS
    }
    raw = feature.raw_components()
    contributions = {
        component: effective_weights[component] * raw[component]
        for component in COMPONENTS
    }
    artifact_contribution = -ARTIFACT_PENALTY_WEIGHT * feature.artifact_penalty
    adjustment = usage_adjustment(guidance)
    final_score = sum(contributions.values()) + artifact_contribution + adjustment
    return {
        "condition_id": condition_id,
        "selection_method": "score_rank",
        "raw_components": raw,
        "enabled_flags": enabled,
        "configured_weights": dict(configured),
        "effective_weights": effective_weights,
        "component_contributions": contributions,
        "artifact_penalty_raw": feature.artifact_penalty,
        "artifact_penalty_weight": ARTIFACT_PENALTY_WEIGHT,
        "artifact_contribution": artifact_contribution,
        "usage_quota_guidance": dict(guidance),
        "usage_quota_contribution": adjustment,
        "final_score": final_score,
        "candidate_id": feature.candidate_id,
        "tie_break_key": feature.candidate_id,
        "sample_seed": sample_seed,
        "attempt": attempt,
        "tone_matching_enabled": True,
        "alpha_blending_enabled": True,
        "boundary_rendering_enabled": True,
    }


def _eligible_unique(features: Iterable[CandidateFeature]) -> list[CandidateFeature]:
    by_id: dict[str, CandidateFeature] = {}
    for feature in features:
        if feature.eligible and feature.candidate_id not in by_id:
            by_id[feature.candidate_id] = feature
    return [by_id[key] for key in sorted(by_id)]


def select_candidate(
    condition_id: str,
    plan: CandidatePlan,
    features: Iterable[CandidateFeature],
    state: UsageState,
    campaign_seed: int,
    *,
    scoring_function: Callable[..., dict[str, Any]] = score_candidate,
) -> tuple[CandidateFeature, dict[str, Any]] | None:
    if condition_id not in CONDITIONS:
        raise ManifestError(f"unsupported condition: {condition_id}")
    eligible = _eligible_unique(features)
    if not eligible:
        return None
    if condition_id == "A0":
        digest, seed = derive_seed(
            campaign_seed,
            plan.sample_id,
            "condition_random_selection",
            plan.attempt,
            condition_id=condition_id,
        )
        selected = random.Random(seed).choice(eligible)
        metadata = {
            "condition_id": condition_id,
            "selection_method": "seeded_random_no_score_ranking",
            "raw_components": selected.raw_components(),
            "enabled_flags": {component: False for component in COMPONENTS},
            "configured_weights": dict(MODE_WEIGHTS[plan.mode]),
            "effective_weights": {component: 0.0 for component in COMPONENTS},
            "component_contributions": {component: 0.0 for component in COMPONENTS},
            "artifact_penalty_raw": selected.artifact_penalty,
            "artifact_penalty_weight": 0.0,
            "artifact_contribution": 0.0,
            "usage_quota_guidance": None,
            "usage_quota_contribution": 0.0,
            "final_score": None,
            "candidate_id": selected.candidate_id,
            "tie_break_key": selected.candidate_id,
            "sample_seed": plan.candidate_seed,
            "condition_selection_seed": digest,
            "attempt": plan.attempt,
            "tone_matching_enabled": True,
            "alpha_blending_enabled": True,
            "boundary_rendering_enabled": True,
        }
        return selected, metadata

    ranked: list[tuple[float, str, CandidateFeature, dict[str, Any]]] = []
    for feature in eligible:
        guidance = state.guidance(
            feature.foreground_id, plan.domain, plan.background_id
        )
        if guidance["remaining_quota"] <= 0:
            continue
        metadata = scoring_function(
            condition_id,
            feature,
            plan.mode,
            guidance,
            sample_seed=plan.candidate_seed,
            attempt=plan.attempt,
        )
        ranked.append(
            (-float(metadata["final_score"]), feature.candidate_id, feature, metadata)
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (item[0], item[1]))
    _, _, selected, metadata = ranked[0]
    return selected, metadata
