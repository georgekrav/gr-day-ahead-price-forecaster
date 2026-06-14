"""Release-gate tests: promotion rule and the health audit trail."""

import json

from gr_epf import governance


class TestEvaluateGate:
    def test_passes_when_model_beats_naive_within_bounds(self):
        result = governance.evaluate_gate(model_mae=16.0, naive_mae=22.0, baseline_mae=None)
        assert result.passed
        assert result.reasons == []

    def test_fails_when_margin_over_naive_too_small(self):
        # 21.0 is below 22.0 but not by the required 10%
        result = governance.evaluate_gate(model_mae=21.0, naive_mae=22.0, baseline_mae=None)
        assert not result.passed
        assert any("beat naive" in r for r in result.reasons)

    def test_fails_on_absolute_ceiling(self):
        # beats naive comfortably but the absolute level is implausible
        result = governance.evaluate_gate(model_mae=45.0, naive_mae=70.0, baseline_mae=None)
        assert not result.passed
        assert any("ceiling" in r for r in result.reasons)

    def test_fails_on_regression_against_baseline(self):
        # beats naive and under ceiling, but 30% worse than last accepted run
        result = governance.evaluate_gate(model_mae=22.0, naive_mae=30.0, baseline_mae=16.0)
        assert not result.passed
        assert any("regress" in r for r in result.reasons)

    def test_first_run_skips_baseline_check(self):
        result = governance.evaluate_gate(model_mae=30.0, naive_mae=40.0, baseline_mae=None)
        assert result.passed

    def test_within_regression_tolerance_passes(self):
        # 20% worse than baseline is inside the 25% default tolerance
        result = governance.evaluate_gate(model_mae=19.2, naive_mae=26.0, baseline_mae=16.0)
        assert result.passed

    def test_config_is_tunable(self):
        strict = governance.GateConfig(min_improvement_vs_naive=0.30)
        result = governance.evaluate_gate(16.0, 22.0, None, config=strict)
        assert not result.passed


class TestHealthRecord:
    def test_baseline_none_without_file(self, tmp_path):
        assert governance.load_baseline_mae(tmp_path / "missing.json") is None

    def test_baseline_reads_last_accepted(self, tmp_path):
        path = tmp_path / "model_health.json"
        governance.append_health_record(path, {"mae": 17.0, "status": "accepted"})
        governance.append_health_record(path, {"mae": 99.0, "status": "rejected"})
        governance.append_health_record(path, {"mae": 16.5, "status": "accepted"})
        assert governance.load_baseline_mae(path) == 16.5

    def test_append_sets_latest_and_grows_history(self, tmp_path):
        path = tmp_path / "model_health.json"
        governance.append_health_record(path, {"mae": 17.0, "status": "accepted"})
        governance.append_health_record(path, {"mae": 16.0, "status": "accepted"})
        record = json.loads(path.read_text())
        assert len(record["history"]) == 2
        assert record["latest"]["mae"] == 16.0
