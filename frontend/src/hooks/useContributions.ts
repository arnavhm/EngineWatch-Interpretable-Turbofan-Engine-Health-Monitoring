import { useState, useEffect } from 'react';
import { getContributions } from '../api';
import type { ContributionsResponse } from '../types';

export function useContributions(engineId: number, datasetId: string) {
  const [data, setData] = useState<ContributionsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!engineId || isNaN(engineId)) return;
    let mounted = true;
    setLoading(true);
    setError(null);
    getContributions(engineId, datasetId)
      .then(res => { if (mounted) setData(res); })
      .catch(err => { console.error('Contributions error:', err); if (mounted) setError(String(err)); })
      .finally(() => { if (mounted) setLoading(false); });
    return () => { mounted = false; };
  }, [engineId, datasetId]);

  return { data, loading, error };
}
