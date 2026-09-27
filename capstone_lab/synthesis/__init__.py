"""Deterministic S3 synthesis adapter and fixture execution boundary."""

from .contract import (
    CONDITIONS,
    CandidateFeature,
    CandidatePlan,
    FixtureConfig,
    derive_seed,
)
from .executor import run_fixture_campaign
from .scoring import UsageState, score_candidate, select_candidate
from .smoke import (
    SmokeConfig,
    build_actual_candidate_plans,
    build_smoke_source_manifest,
    run_actual_smoke_campaign,
)

__all__ = [
    "CONDITIONS",
    "CandidateFeature",
    "CandidatePlan",
    "FixtureConfig",
    "UsageState",
    "SmokeConfig",
    "build_actual_candidate_plans",
    "build_smoke_source_manifest",
    "derive_seed",
    "run_actual_smoke_campaign",
    "run_fixture_campaign",
    "score_candidate",
    "select_candidate",
]
