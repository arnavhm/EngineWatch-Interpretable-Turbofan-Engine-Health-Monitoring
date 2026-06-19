import { useState, useEffect } from 'react';
import { getTopRisk } from '../api';
import type { TopRiskItem } from '../types';

export function useTopRisk(datasetId: string) {
  const [data, setData] = useState<TopRiskItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) return;

    let mounted = true;
    setLoading(true);
    setError(null);

    getTopRisk(datasetId)
      .then((res) => {
        if (mounted) setData(res);
      })
      .catch((err) => {
        console.error("TopRisk error:", err);
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
