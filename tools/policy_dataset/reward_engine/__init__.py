"""Offline calibrated rewards; no learned selector or hardware implementation."""
from .engine import collect_candidates, evaluate_group, validate_config

__all__ = ["collect_candidates", "evaluate_group", "validate_config"]
