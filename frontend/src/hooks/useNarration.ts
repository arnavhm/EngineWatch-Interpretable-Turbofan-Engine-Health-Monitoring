import { useState, useEffect, useCallback } from 'react';
import { narrateChat } from '../api';

export function useNarration(engineId: number | null, datasetId: string) {
  const [sessionId, setSessionId] = useState<string>('');
  const [messages, setMessages] = useState<Array<{role: string, content: string}>>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [narrationAvailable, setNarrationAvailable] = useState<boolean>(true);

  // Reset session when engine/dataset changes
  useEffect(() => {
    if (engineId === null) return;
    
    // Generate UUID or fallback
    const newSessionId = typeof crypto !== 'undefined' && crypto.randomUUID 
      ? crypto.randomUUID() 
      : Math.random().toString(36).substring(2) + Date.now().toString(36);
      
    setSessionId(newSessionId);
    setMessages([]);
    setNarrationAvailable(true);
    setError(null);
    setLoading(true);
    
    let mounted = true;
    
    narrateChat({ dataset_id: datasetId, engine_id: engineId, session_id: newSessionId })
      .then(res => {
        if (!mounted) return;
        setNarrationAvailable(res.narration_available);
        if (res.reply) {
          setMessages([{ role: 'assistant', content: res.reply }]);
        }
      })
      .catch(err => {
        if (!mounted) return;
        console.error('Narration init error:', err);
        setError(String(err));
      })
      .finally(() => {
        if (mounted) setLoading(false);
      });
      
    return () => { mounted = false; };
  }, [engineId, datasetId]);

  const sendMessage = useCallback(async (text: string) => {
    if (!engineId || !sessionId || !text.trim()) return;
    
    const userMessage = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setLoading(true);
    setError(null);
    
    try {
      const res = await narrateChat({
        dataset_id: datasetId,
        engine_id: engineId,
        session_id: sessionId,
        message: text
      });
      
      setNarrationAvailable(res.narration_available);
      if (res.reply) {
        setMessages(prev => [...prev, { role: 'assistant', content: res.reply as string }]);
      }
    } catch (err) {
      console.error('Narration send error:', err);
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [engineId, datasetId, sessionId]);

  return { messages, loading, error, narrationAvailable, sendMessage };
}
