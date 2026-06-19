import { useState, useEffect } from 'react';
import { getFleetSummary } from '../api';
import type { FleetSummary } from '../types';

export function useFleetSummary(datasetId: string) {
  const [data, setData] = useState<FleetSummary | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) return;

    let mounted = true;
    setLoading(true);
    setError(null);

    getFleetSummary(datasetId)
      .then((res) => {
        if (mounted) setData(res);
      })
      .catch((err) => {
        console.error("FleetSummary error:", err);
        if (mounted) setError(String(err));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, [datasetId]);

  return { data, loading, error };
}
