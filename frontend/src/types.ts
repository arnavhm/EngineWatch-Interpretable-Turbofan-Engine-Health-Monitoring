export type RiskState = "Healthy" | "Degrading" | "Critical";

export interface HistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface TrendDecile {
  life_pct_bin: number;
  mean_risk_score: number;
  n_engines_contributing: number;
}

export interface FleetAnalyticsResponse {
  dataset_id: string;
  risk_histogram: HistogramBin[];
  state_counts: Partial<Record<RiskState, number>>;
  risk_trend: TrendDecile[];
}

export interface FleetCompareRow {
  dataset_id: string;
  fleet_size: number;
  state_counts: Partial<Record<RiskState, number>>;
  n_critical: number;
  mean_rul: number;
  median_rul: number;
}

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

export interface TrajectoryResponse {
  engine_id: number;
  dataset_id: string;
  cycles: number[];
  health_index: number[];
  velocity: number[];
  variability: number[];
}

export interface SensorMetadata {
  values: number[];
  descriptive_name: string;
  layman_text: string;
  explanation: string;
  units: string;
  signal_direction: string;
  confirmed: boolean;
  module: string;
}

export interface SensorResponse {
  engine_id: number;
  dataset_id: string;
  cycles: number[];
  sensors: Record<string, SensorMetadata>;
}

export interface AnomalyPoint {
  engine_id: number;
  health_index: number;
  velocity: number;
  is_anomaly: boolean;
  risk_state: RiskState;
}

export interface CsvPrediction {
  engine_id: number;
  risk_score: number;
  rul_cycles: number;
  risk_state: string;
  health_index?: number;
}


// ─────────────────────────────────────────────────────────────────────────────
// ADDITIONS FOR src/types.ts
// Append after your existing PredictResponse / FleetSummary interfaces.
// ─────────────────────────────────────────────────────────────────────────────

export interface SensorContribution {
  sensor_id: string;         // "s11"
  symbol: string;            // "Ps30"
  description: string;       // "Static pressure at HPC outlet"
  signed_contribution: number;
  abs_contribution: number;
}

export interface ModuleHeat {
  module: string;                    // "hpc"
  display_name: string;             // "HPC"
  direction: "healthy" | "critical" | "inactive";
  signed_heat: number;
  norm_magnitude: number;           // [0,1] — dominant module = 1.0
  norm_signed: number;              // [-1,1]
  active_sensors: SensorContribution[];
  is_active: boolean;
}

export interface ContributionsResponse {
  engine_id: number;
  dataset_id: string;
  cycle: number;
  dominant_module: string;
  dominant_driver_text: string;     // "HPC — Ps30, T30, phi driving degradation"
  modules: ModuleHeat[];
}

export interface NarrationRequest {
  dataset_id: string;
  engine_id: number;
  session_id: string;
  message?: string;
}

export interface NarrationResponse {
  session_id: string;
  reply: string | null;
  narration_available: boolean;
  role_used: string | null;
}
