export type RiskState = "Healthy" | "Degrading" | "Critical";

export interface PredictResponse {
  engine_id: number;
  dataset_id: string;
  health_index: number;
  risk_score: number;
  risk_state: RiskState;
  rul_cycles: number;
  ci_lower: number;
  ci_upper: number;
  ci_std: number;
  model_name: string;
  rmse: number;
}

export interface FleetSummary {
  dataset_id: string;
  n_engines: number;
  state_counts: Partial<Record<RiskState, number>>; // a state can be absent if count is 0
  n_critical: number;
  mean_rul: number;
  median_rul: number;
  highest_risk_engine: number;
}

export interface TopRiskItem {
  engine_id: number;
  risk_score: number;
  risk_state: RiskState;
  rul_cycles: number;
}
