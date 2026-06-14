"""Monthly release gate: the champion/challenger promotion rule.

Context that shapes this module: the daily job (scripts/make_forecast.py)
retrains LightGBM from scratch on the full history every morning, so the
model weights are never stale. What is *not* refreshed daily are the
conformal calibration widths (forecasts/calibration.parquet) and the
published accuracy metrics (forecasts/backtest_summary.json) -- both come
from a walk-forward backtest that, until now, was run by hand once. The
monthly pipeline recomputes them on the most recent data and must clear
this gate before the refreshed ("challenger") artifacts replace the live
("champion") ones.

A failed gate is a halt, not a fallback: the pipeline keeps the last-good
artifacts and exits non-zero so CI emails a human, rather than silently
shipping a calibration built on a bad month of data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GateConfig:
    # The model exists to beat the day-ahead naive baseline; if a fresh
    # backtest no longer shows a clear margin, something upstream broke.
    min_improvement_vs_naive: float = 0.10
    # Absolute ceiling catches blow-ups (a corrupted month, an ENTSO-E
    # schema change) that might still beat an equally broken naive.
    max_abs_mae: float = 40.0
    # Guard against silent drift: reject if MAE regresses sharply against
    # the last accepted run. Skipped on the first run (no baseline yet).
    max_regression_vs_baseline: float = 0.25


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reasons: list[str]


# Frozen, so a shared default instance is safe to reference from signatures.
DEFAULT_GATE = GateConfig()


def evaluate_gate(
    model_mae: float,
    naive_mae: float,
    baseline_mae: float | None,
    config: GateConfig = DEFAULT_GATE,
) -> GateResult:
    """Decide whether a fresh backtest is healthy enough to promote.

    baseline_mae is the MAE of the last accepted monthly run, or None on the
    very first run. All checks must pass; every failure is collected so the
    CI log explains exactly why a promotion was blocked.
    """
    reasons: list[str] = []
    margin = config.min_improvement_vs_naive
    naive_target = naive_mae * (1 - margin)
    if model_mae > naive_target:
        reasons.append(
            f"MAE {model_mae:.2f} does not beat naive {naive_mae:.2f} by "
            f"{margin:.0%} (need <= {naive_target:.2f})"
        )
    if model_mae > config.max_abs_mae:
        reasons.append(
            f"MAE {model_mae:.2f} exceeds absolute ceiling {config.max_abs_mae:.2f}"
        )
    if baseline_mae is not None:
        ceiling = baseline_mae * (1 + config.max_regression_vs_baseline)
        if model_mae > ceiling:
            reasons.append(
                f"MAE {model_mae:.2f} regresses past "
                f"{config.max_regression_vs_baseline:.0%} over baseline "
                f"{baseline_mae:.2f} (limit {ceiling:.2f})"
            )
    return GateResult(passed=not reasons, reasons=reasons)


def load_baseline_mae(health_path: Path) -> float | None:
    """MAE of the most recent accepted run, or None if there is no history."""
    if not health_path.exists():
        return None
    record = json.loads(health_path.read_text())
    accepted = [r for r in record.get("history", []) if r.get("status") == "accepted"]
    return accepted[-1]["mae"] if accepted else None


def append_health_record(health_path: Path, entry: dict) -> None:
    """Append one run to the committed audit trail and mark it as latest."""
    record = json.loads(health_path.read_text()) if health_path.exists() else {}
    history = record.get("history", [])
    history.append(entry)
    record["history"] = history
    record["latest"] = entry
    health_path.parent.mkdir(parents=True, exist_ok=True)
    health_path.write_text(json.dumps(record, indent=2) + "\n")
