import { useState, useEffect } from 'react';
import { getFleetAnalytics } from '../api';
import type { FleetAnalyticsResponse } from '../types';

export function useFleetAnalytics(datasetId: string) {
  const [data, setData] = useState<FleetAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) return;

    let mounted = true;
    setLoading(true);
    setError(null);

    getFleetAnalytics(datasetId)
      .then((res) => {
        if (mounted) setData(res);
      })
      .catch((err) => {
        console.error("FleetAnalytics error:", err);
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
