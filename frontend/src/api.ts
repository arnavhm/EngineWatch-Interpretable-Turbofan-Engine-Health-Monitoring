import type { PredictResponse, FleetSummary, TopRiskItem } from './types';

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