import type { AnomalyPoint } from '../types';

interface EngineSelectorProps {
  points: AnomalyPoint[] | null;
  loading: boolean;
  error: string | null;
  onSelectEngine: (id: number) => void;
}

export default function EngineSelector({ points, loading, error, onSelectEngine }: EngineSelectorProps) {
  const engines = points ? [...points].sort((a, b) => a.engine_id - b.engine_id) : [];
  const disabled = loading || !!error || engines.length === 0;

  const placeholder = loading
    ? 'Loading fleet…'
    : error
      ? 'Fleet list unavailable'
      : `Open an engine (${engines.length} in fleet)…`;

  return (
    <div className="flex flex-col w-full sm:flex-1">
      <label
        htmlFor="engine-unit-select"
        className="text-[10px] text-muted mb-1.5 font-mono uppercase tracking-widest font-semibold"
      >
        Engine Unit
      </label>
      <select
        id="engine-unit-select"
        value=""
        disabled={disabled}
        onChange={(e) => {
          if (e.target.value === '') return;
          onSelectEngine(Number(e.target.value));
        }}
        className="bg-panel2 border border-border rounded-md text-text font-mono px-3 py-2 outline-none focus:border-accent w-full transition-colors shadow-sm disabled:opacity-50"
      >
        <option value="" disabled>
          {placeholder}
        </option>
        {engines.map((p) => (
          <option key={p.engine_id} value={p.engine_id}>
            Engine {p.engine_id} — {p.risk_state}
          </option>
        ))}
      </select>
    </div>
  );
}
