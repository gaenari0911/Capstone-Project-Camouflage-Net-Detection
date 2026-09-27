"""Strict S5 training, checkpoint, metric, and scheduling adapters."""

from .metrics import evaluate_paired_masks
from .scheduler import run_s5_smoke_campaign

__all__ = ["evaluate_paired_masks", "run_s5_smoke_campaign"]
