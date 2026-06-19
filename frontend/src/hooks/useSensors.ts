import { useState, useEffect } from 'react';
import { getSensors } from '../api';
import type { SensorResponse } from '../types';

export function useSensors(engineId: number, datasetId: string) {
  const [data, setData] = useState<SensorResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!engineId || isNaN(engineId)) return;
    let mounted = true;
    setLoading(true);
    setError(null);
    getSensors(engineId, datasetId)
      .then(res => { if (mounted) setData(res); })
      .catch(err => { console.error('Sensors error:', err); if (mounted) setError(String(err)); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [engineId, datasetId]);

  return { data, loading, error };
}
