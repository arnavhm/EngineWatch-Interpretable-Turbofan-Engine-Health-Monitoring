import copy
import json
from typing import Any, Callable, Optional

import pandas as pd

from app.components.aog_cost_simulator import compute_maintenance_decision


def get_agentic_tools(
    engine_context: dict[str, Any],
    config: dict[str, Any],
    predicted_rul: float,
    fleet_df: Optional[pd.DataFrame],
) -> list[Callable]:
    """
    Returns a list of callable tools bound to the current engine and fleet context.
    These tools are passed to the Gemini SDK to enable function calling.
    """

    def simulate_aog_cost(
        revenue_loss_per_day_rs_lakh: float = None,
        aog_cost_per_event_rs_cr: float = None,
    ) -> str:
        """
        Simulates the Expected AOG (Aircraft On Ground) cost under different financial assumptions for the current engine.
        Use this tool when the user asks "what if" questions about changing revenue loss or AOG costs.

        Args:
            revenue_loss_per_day_rs_lakh: Optional override for the daily revenue loss in Rs Lakhs.
            aog_cost_per_event_rs_cr: Optional override for the direct AOG cost per event in Rs Crores.

        Returns:
            A JSON string containing the simulated costs and the new maintenance decision.
        """
        # copy config to avoid mutating global state
        sim_config = copy.deepcopy(config)
        if revenue_loss_per_day_rs_lakh is not None:
            sim_config["aog_simulator"]["revenue_per_day_rs_lakh"] = float(
                revenue_loss_per_day_rs_lakh
            )
        if aog_cost_per_event_rs_cr is not None:
            sim_config["aog_simulator"]["aog_cost_per_event_rs_cr"] = float(
                aog_cost_per_event_rs_cr
            )

        try:
            decision = compute_maintenance_decision(
                risk_score=engine_context["risk_score"],
                rul_cycles=int(predicted_rul),
                risk_state=engine_context["risk_state"],
                config=sim_config,
            )
            return json.dumps(decision, indent=2)
        except Exception as e:
            return f"Error running simulation: {e}"

    def query_fleet_status(sort_by: str = "risk_score", top_n: int = 5) -> str:
        """
        Queries the current status of the entire engine fleet. Use this tool when the user asks about other engines, the overall fleet, or which engines are at highest risk.

        Args:
            sort_by: The column to sort by. Allowed values: 'risk_score', 'health_index', 'HI_velocity', 'HI_variability', 'cycle'. Defaults to 'risk_score'.
            top_n: The number of top engines to return. Defaults to 5. Maximum is 10.

        Returns:
            A JSON string containing the top N engines and their current metrics.
        """
        if fleet_df is None or fleet_df.empty:
            return "Fleet data is currently unavailable."

        allowed_sort = {
            "risk_score",
            "health_index",
            "HI_velocity",
            "HI_variability",
            "cycle",
        }
        if sort_by not in allowed_sort:
            sort_by = "risk_score"

        # Bound top_n
        top_n = min(max(1, int(top_n)), 10)

        # Get the latest cycle for each engine in case fleet_df has history
        latest_fleet = fleet_df.sort_values("cycle").groupby("unit").tail(1)

        if sort_by not in latest_fleet.columns:
            return f"Error: Metric '{sort_by}' is not available in the fleet dataset."

        # For health_index, lower is worse, so we sort ascending. For others, higher is worse, sort descending.
        ascending = sort_by in {"health_index"}
        sorted_fleet = latest_fleet.sort_values(sort_by, ascending=ascending).head(
            top_n
        )

        cols = [
            "unit",
            "cycle",
            "risk_state",
            "risk_score",
            "health_index",
            "HI_velocity",
            "HI_variability",
        ]
        cols = [c for c in cols if c in sorted_fleet.columns]

        results = sorted_fleet[cols].to_dict(orient="records")
        return json.dumps(results, indent=2)

    return [simulate_aog_cost, query_fleet_status]
