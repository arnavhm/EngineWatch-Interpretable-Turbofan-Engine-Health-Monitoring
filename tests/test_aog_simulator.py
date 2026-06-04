import pytest
from data.load import load_config
from app.components.aog_cost_simulator import (
    compute_failure_probability,
    compute_aog_expected_cost,
    compute_maintenance_decision,
)


@pytest.fixture
def config():
    return load_config()


# ---------------------------------------------------------------------------
# Scenario tests — manual validation cases from April 21, 2026
# ---------------------------------------------------------------------------


def test_scenario_1_healthy_low_risk(config):
    """risk=0.10, RUL=150, Healthy → LOW urgency, no immediate action."""
    result = compute_maintenance_decision(0.10, 150, "Healthy", config)
    assert result["urgency_level"] == "LOW"
    assert result["act_now"] is False
    assert "no immediate action" in result["recommendation"].lower()


def test_scenario_2_degrading_moderate(config):
    """risk=0.50, RUL=60, Degrading → MODERATE or HIGH urgency, scheduling language."""
    result = compute_maintenance_decision(0.50, 60, "Degrading", config)
    assert result["urgency_level"] in ("MODERATE", "HIGH")
    rec_lower = result["recommendation"].lower()
    assert "schedule" in rec_lower or "inspection" in rec_lower


def test_scenario_3_critical_high_risk(config):
    """risk=0.85, RUL=20, Critical → HIGH or CRITICAL urgency, maintenance language."""
    result = compute_maintenance_decision(0.85, 20, "Critical", config)
    assert result["urgency_level"] in ("HIGH", "CRITICAL")
    rec_lower = result["recommendation"].lower()
    assert "preventive" in rec_lower or "maintenance" in rec_lower or "inspection" in rec_lower


def test_scenario_4_critical_act_now(config):
    """risk=0.95, RUL=8, Critical → CRITICAL urgency, act_now=True."""
    result = compute_maintenance_decision(0.95, 8, "Critical", config)
    assert result["urgency_level"] == "CRITICAL"
    assert result["act_now"] is True


def test_scenario_5_critical_near_failure(config):
    """risk=0.99, RUL=2, Critical → CRITICAL, act_now=True, high failure probability."""
    result = compute_maintenance_decision(0.99, 2, "Critical", config)
    assert result["urgency_level"] == "CRITICAL"
    assert result["act_now"] is True
    assert result["failure_probability"] >= 0.95


# ---------------------------------------------------------------------------
# Structural tests — compute_failure_probability
# ---------------------------------------------------------------------------


def test_failure_probability_at_zero_risk(config):
    prob = compute_failure_probability(0.0, config)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_failure_probability_at_full_risk(config):
    prob = compute_failure_probability(1.0, config)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_failure_probability_monotonic(config):
    """Higher risk score must yield strictly higher failure probability."""
    prob_low = compute_failure_probability(0.1, config)
    prob_high = compute_failure_probability(0.5, config)
    assert prob_high > prob_low, (
        f"Expected P(0.5) > P(0.1) but got {prob_high:.4f} <= {prob_low:.4f}"
    )


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_failure_probability_raises_for_risk_above_one(config):
    with pytest.raises(ValueError):
        compute_failure_probability(1.5, config)


def test_maintenance_decision_raises_for_invalid_risk_state(config):
    with pytest.raises(ValueError):
        compute_maintenance_decision(0.5, 60, "Unknown", config)


def test_maintenance_decision_raises_for_risk_above_one(config):
    with pytest.raises(ValueError):
        compute_maintenance_decision(1.5, 60, "Degrading", config)
