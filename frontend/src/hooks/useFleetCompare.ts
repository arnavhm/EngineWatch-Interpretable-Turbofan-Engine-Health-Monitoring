import { useState, useEffect } from 'react';
import { getFleetCompare } from '../api';
import type { FleetCompareRow } from '../types';

export function useFleetCompare() {
  const [data, setData] = useState<FleetCompareRow[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;
    setLoading(true);
    setError(null);

    getFleetCompare()
      .then((res) => {
        if (mounted) setData(res);
      })
      .catch((err) => {
        console.error("FleetCompare error:", err);
        if (mounted) setError(String(err));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });

    return () => {
      mounted = false;
    };
  }, []);

  return { data, loading, error };
}
