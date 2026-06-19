import type { PredictResponse, FleetSummary, TopRiskItem } from './types';

const BASE = import.meta.env.VITE_API_BASE;

async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export async function getPredict(engineId: number, datasetId: string): Promise<PredictResponse> {
  const params = new URLSearchParams({ engine_id: String(engineId), dataset_id: datasetId });
  return fetchJson<PredictResponse>(`/predict?${params.toString()}`);
}

export async function getFleetSummary(datasetId: string): Promise<FleetSummary> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  return fetchJson<FleetSummary>(`/fleet/summary?${params.toString()}`);
}

export async function getTopRisk(datasetId: string): Promise<TopRiskItem[]> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  return fetchJson<TopRiskItem[]>(`/fleet/top-risk?${params.toString()}`);
}
