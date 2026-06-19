import { useState, useEffect } from 'react';
import { getPredict } from '../api';
import type { PredictResponse } from '../types';

export function usePredict(engineId: number, datasetId: string) {
  const [data, setData] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!engineId || isNaN(engineId)) return;
    
    let mounted = true;
    setLoading(true);
    setError(null);
    
    getPredict(engineId, datasetId)
      .then(res => {
        if (mounted) setData(res);
      })
      .catch(err => {
        console.error("Predict error:", err);
        if (mounted) setError(String(err));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
      
    return () => { mounted = false; };
  }, [engineId, datasetId]);

  return { data, loading, error };
}
