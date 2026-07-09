import { useState, useEffect } from 'react';
import { narrateChat } from '../api';

export function useNarration(engineId: number | null, datasetId: string) {
  const [reply, setReply] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [narrationAvailable, setNarrationAvailable] = useState<boolean>(true);

  useEffect(() => {
    if (engineId === null) return;
    const sessionId = typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).substring(2) + Date.now().toString(36);

    setReply(null);
    setNarrationAvailable(true);
    setError(null);
    setLoading(true);
    let mounted = true;

    narrateChat({ dataset_id: datasetId, engine_id: engineId, session_id: sessionId })
      .then((res) => { if (mounted) { setNarrationAvailable(res.narration_available); setReply(res.reply); } })
      .catch((err) => { if (mounted) { console.error('Narration fetch error:', err); setError(String(err)); } })
      .finally(() => { if (mounted) setLoading(false); });

    return () => { mounted = false; };
  }, [engineId, datasetId]);

  return { reply, loading, error, narrationAvailable };
}
