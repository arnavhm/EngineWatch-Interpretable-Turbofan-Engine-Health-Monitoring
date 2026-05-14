"""
AOG Cost Impact Simulator - Engine Watch Iteration 2
Pure Calculation Layer. NO ML. No Retraining.
Reads pipeline outputs and produces an economic decision.
All cost paramaters from config/config.yaml -> aog_simulator section.
"""

import numpy as np


def compute_failure_probability(risk_score: float, config: dict) -> float:
    """
    Purpose:      Convert risk_score [0,1] to P(failure) via sigmoid.
                  Calibrated to FD001 cluster centroids:
                    Healthy (0.46)   → P ≈ 0.15
                    Degrading (0.66) → P ≈ 0.50
                    Critical (0.87)  → P ≈ 0.85
    Input:        risk_score float in [0,1], config dict
    Output:       float in [0,1]
    Failure:      ValueError if risk_score outside [0,1]
    """
    if not 0.0 <= risk_score <= 1.0:
        raise ValueError(f"risk_score must be in [0,1], got {risk_score:.4f}")

    midpoint: float = config["aog_simulator"]["sigmoid_midpoint"]
    steepness: float = config["aog_simulator"]["sigmoid_steepness"]

    prob = 1.0 / (1.0 + np.exp(-steepness * (risk_score - midpoint)))
    return float(np.clip(prob, 0.0, 1.0))


def compute_aog_expected_cost(
    failure_prob: float,
    rul_cycles: int,
    config: dict,
) -> dict:
    """
    Purpose:      Compute scenario costs for preventive vs reactive maintenance.
    Input:        failure_prob float [0,1], rul_cycles int >= 0, config dict
    Output:       dict — see keys below
    Assumptions:  1 cycle = 1 flight. All monetary values in Rs Crores.
    Failure:      ValueError if failure_prob outside [0,1]
    """
    if not 0.0 <= failure_prob <= 1.0:
        raise ValueError(f"failure_prob must be in [0,1], got {failure_prob:.4f}")

    cfg: dict = config["aog_simulator"]

    flights_per_day: float = cfg["flights_per_day"]
    aog_duration_days: float = cfg["expected_aog_duration_days"]
    failure_window_days: float = rul_cycles / flights_per_day

    # Scenario A — Preventive
    preventive_cost_rs_cr: float = cfg["preventive_maintenance_cost_rs_cr"]

    # Scenario B — Reactive
    aog_direct_cost_rs_cr: float = cfg["aog_cost_per_event_rs_cr"]

    revenue_loss_rs_cr: float = (
        cfg["revenue_per_day_rs_lakh"] * aog_duration_days / 100.0
    )

    disruption_cost_rs_cr: float = (
        cfg["disruption_cost_per_passenger_rs"]
        * cfg["passengers_per_flight"]
        * flights_per_day
        * aog_duration_days
        / 1e7
    )

    total_reactive_cost_rs_cr: float = (
        aog_direct_cost_rs_cr + revenue_loss_rs_cr + disruption_cost_rs_cr
    )

    expected_aog_cost_rs_cr: float = failure_prob * total_reactive_cost_rs_cr
    estimated_saving_rs_cr: float = expected_aog_cost_rs_cr - preventive_cost_rs_cr

    return {
        "failure_window_days": round(failure_window_days, 1),
        "preventive_cost_rs_cr": round(preventive_cost_rs_cr, 2),
        "aog_direct_cost_rs_cr": round(aog_direct_cost_rs_cr, 2),
        "revenue_loss_rs_cr": round(revenue_loss_rs_cr, 2),
        "disruption_cost_rs_cr": round(disruption_cost_rs_cr, 2),
        "total_reactive_cost_rs_cr": round(total_reactive_cost_rs_cr, 2),
        "expected_aog_cost_rs_cr": round(expected_aog_cost_rs_cr, 2),
        "estimated_saving_rs_cr": round(estimated_saving_rs_cr, 2),
    }


def compute_maintenance_decision(
    risk_score: float,
    rul_cycles: int,
    risk_state: str,
    config: dict,
) -> dict:
    """
    Purpose:      Main entry point. Chains failure probability + cost calculation.
                  Returns complete decision package for dashboard rendering.
    Input:        risk_score float [0,1], rul_cycles int,
                  risk_state str Healthy|Degrading|Critical, config dict
    Output:       dict with recommendation, urgency, costs, act_now, explanation
    Failure:      ValueError on invalid inputs. KeyError if aog_simulator missing.
    """
    valid_states = {"Healthy", "Degrading", "Critical"}
    if risk_state not in valid_states:
        raise ValueError(
            f"risk_state must be one of {valid_states}, got '{risk_state}'"
        )
    if rul_cycles < 0:
        raise ValueError(f"rul_cycles must be >= 0, got {rul_cycles}")

    failure_prob: float = compute_failure_probability(risk_score, config)
    costs: dict = compute_aog_expected_cost(failure_prob, rul_cycles, config)

    act_now: bool = costs["estimated_saving_rs_cr"] > 0.0

    # Urgency thresholds — upper bound from Critical centroid (0.87)
    if risk_score >= 0.85 or rul_cycles <= 15:
        urgency_level = "CRITICAL"
        recommendation = "IMMEDIATE ACTION — Ground for inspection before next flight"
    elif risk_score >= 0.70 or rul_cycles <= 40:
        urgency_level = "HIGH"
        recommendation = "Schedule preventive maintenance within 3 flight cycles"
    elif risk_score >= 0.55 or rul_cycles <= 80:
        urgency_level = "MODERATE"
        recommendation = "Schedule inspection at next available maintenance slot"
    else:
        urgency_level = "LOW"
        recommendation = "Continue monitoring — no immediate action required"

    # Plain English explanation for dashboard expander
    fp_pct = f"{failure_prob:.0%}"
    if act_now:
        explanation = (
            f"Failure probability estimated at {fp_pct} with {rul_cycles} cycles "
            f"remaining (~{costs['failure_window_days']} days). "
            f"Expected AOG cost (Rs {costs['expected_aog_cost_rs_cr']:.1f} Cr) "
            f"exceeds preventive cost (Rs {costs['preventive_cost_rs_cr']:.1f} Cr) "
            f"by Rs {costs['estimated_saving_rs_cr']:.1f} Cr. "
            f"Preventive action is economically justified."
        )
    else:
        explanation = (
            f"Failure probability estimated at {fp_pct} with {rul_cycles} cycles "
            f"remaining (~{costs['failure_window_days']} days). "
            f"Expected AOG cost (Rs {costs['expected_aog_cost_rs_cr']:.1f} Cr) "
            f"does not yet exceed preventive cost "
            f"(Rs {costs['preventive_cost_rs_cr']:.1f} Cr). Continue monitoring."
        )

    return {
        "recommendation": recommendation,
        "urgency_level": urgency_level,
        "failure_probability": round(failure_prob, 3),
        "failure_window_days": costs["failure_window_days"],
        "preventive_cost_rs_cr": costs["preventive_cost_rs_cr"],
        "aog_direct_cost_rs_cr": costs["aog_direct_cost_rs_cr"],
        "revenue_loss_rs_cr": costs["revenue_loss_rs_cr"],
        "disruption_cost_rs_cr": costs["disruption_cost_rs_cr"],
        "total_reactive_cost_rs_cr": costs["total_reactive_cost_rs_cr"],
        "expected_aog_cost_rs_cr": costs["expected_aog_cost_rs_cr"],
        "estimated_saving_rs_cr": costs["estimated_saving_rs_cr"],
        "act_now": act_now,
        "explanation": explanation,
    }
