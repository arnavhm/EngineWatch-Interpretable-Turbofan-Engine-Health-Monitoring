import { useState, useEffect } from 'react';
import { getAnomaly } from '../api';
import type { AnomalyPoint } from '../types';

export function useAnomaly(datasetId: string) {
  const [data, setData] = useState<AnomalyPoint[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);
    getAnomaly(datasetId)
      .then(res => { if (mounted) setData(res); })
      .catch(err => { console.error('Anomaly error:', err); if (mounted) setError(String(err)); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [datasetId]);

  return { data, loading, error };
}
