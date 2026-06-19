import { useState, useEffect } from 'react';
import { getTrajectory } from '../api';
import type { TrajectoryResponse } from '../types';

export function useTrajectory(engineId: number, datasetId: string) {
  const [data, setData] = useState<TrajectoryResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!engineId || isNaN(engineId)) return;
    
    let mounted = true;
    setLoading(true);
    setError(null);
    
    getTrajectory(engineId, datasetId)
      .then(res => {
        if (mounted) setData(res);
      })
      .catch(err => {
        console.error("Trajectory error:", err);
        if (mounted) setError(String(err));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
      
    return () => { mounted = false; };
  }, [engineId, datasetId]);

  return { data, loading, error };
}
