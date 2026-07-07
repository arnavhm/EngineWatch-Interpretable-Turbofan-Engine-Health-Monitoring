import type { PredictResponse, FleetSummary, TopRiskItem, TrajectoryResponse, SensorResponse, AnomalyPoint, ContributionsResponse, FleetAnalyticsResponse, FleetCompareRow } from './types';

const BASE = import.meta.env.VITE_API_BASE;

async function fetchJson<T>(path: string): Promise<T> {
  try {
    const url = `${BASE}${path}`;
    console.log(`Fetching: ${url}`);
    
    const res = await fetch(url);
    
    if (!res || typeof res.ok === 'undefined') {
      throw new Error('Invalid fetch response object');
    }
    
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`API error: ${res.status} ${res.statusText} - ${text}`);
    }
    
    return await res.json();
  } catch (error) {
    console.error('fetchJson error:', error);
    throw error;
  }
}

export async function checkHealth(): Promise<boolean> {
  try {
    const url = `${BASE}/health`;
    console.log(`Health check: ${url}`);
    const res = await fetch(url);
    console.log('Health response:', res);
    return res?.ok ?? false;
  } catch (error) {
    console.error('Health check failed:', error);
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

export async function getTrajectory(engineId: number, datasetId: string): Promise<TrajectoryResponse> {
  const params = new URLSearchParams({ engine_id: String(engineId), dataset_id: datasetId });
  return fetchJson<TrajectoryResponse>(`/trajectory?${params.toString()}`);
}

export async function getSensors(engineId: number, datasetId: string): Promise<SensorResponse> {
  const params = new URLSearchParams({ engine_id: String(engineId), dataset_id: datasetId });
  return fetchJson<SensorResponse>(`/sensors?${params.toString()}`);
}

export async function getAnomaly(datasetId: string): Promise<AnomalyPoint[]> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  return fetchJson<AnomalyPoint[]>(`/anomaly?${params.toString()}`);
}

// ADDITION FOR src/api.ts
// Append after your existing getPredict / getFleetSummary functions.
// ─────────────────────────────────────────────────────────────────────────────

// import { ContributionsResponse } from "./types";   // already in types.ts

export async function getFleetAnalytics(datasetId: string): Promise<FleetAnalyticsResponse> {
  const params = new URLSearchParams({ dataset_id: datasetId });
  return fetchJson<FleetAnalyticsResponse>(`/fleet/analytics?${params.toString()}`);
}

export async function getFleetCompare(): Promise<FleetCompareRow[]> {
  return fetchJson<FleetCompareRow[]>(`/fleet/compare`);
}

export async function getContributions(
  engineId: number,
  datasetId: string = "FD001",
): Promise<ContributionsResponse> {
  const res = await fetch(
    `/api/predict/${engineId}/contributions?dataset_id=${datasetId}`,
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

export async function narrateChat(
  req: { dataset_id: string; engine_id: number; session_id: string; message?: string }
) {
  const url = `${BASE}/narrate/chat`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req)
  });
  
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error: ${res.status} - ${text}`);
  }
  return await res.json();
}
