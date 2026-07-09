import { useNarration } from '../hooks/useNarration';

export default function AgentDiagnostic({ engineId, datasetId }: { engineId: number; datasetId: string }) {
  const { reply, loading, error, narrationAvailable } = useNarration(engineId, datasetId);

  if (!narrationAvailable) {
    return (
      <div className="text-muted text-sm font-mono border border-dashed border-border rounded-lg bg-panel2 bg-opacity-50 p-6 text-center">
        <p>Narration unavailable.</p>
        <p className="mt-2 text-xs">Ensure GEMINI_API_KEY is configured in the environment.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {error && <div className="bg-[#E0533A]/10 border border-[#E0533A]/30 text-[#E0533A] text-xs p-2 rounded">Error: {error}</div>}
      {loading && <div className="text-muted text-sm italic">Generating diagnostic…</div>}
      {reply && <p className="text-sm leading-relaxed text-text whitespace-pre-wrap">{reply}</p>}
    </div>
  );
}
