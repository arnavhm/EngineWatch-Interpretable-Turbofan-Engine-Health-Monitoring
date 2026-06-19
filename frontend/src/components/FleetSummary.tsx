import { useFleetSummary } from '../hooks/useFleetSummary';
import { STATE_VAR, STATE_TEXT } from '../stateColors';

interface FleetSummaryPanelProps {
  datasetId: string;
  onSelectEngine: (id: number) => void;
}

export default function FleetSummaryPanel({ datasetId, onSelectEngine }: FleetSummaryPanelProps) {
  const { data, loading, error } = useFleetSummary(datasetId);

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[150px]">
        <span className="text-muted text-sm border-l-2 border-critical pl-4 py-2">Error: {error}</span>
      </div>
    );
  }

  if (loading || !data) {
    return (
      <div className="flex-1 flex flex-col justify-between h-full animate-pulse gap-4 mt-2">
        <div className="w-full h-6 bg-panel2 rounded-full"></div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="h-16 bg-panel2 rounded-md"></div>
          <div className="h-16 bg-panel2 rounded-md"></div>
          <div className="h-16 bg-panel2 rounded-md"></div>
          <div className="h-16 bg-panel2 rounded-md"></div>
        </div>
        <div className="w-1/2 h-4 bg-panel2 rounded-md"></div>
      </div>
    );
  }

  const {
    n_engines,
    mean_rul,
    median_rul,
    n_critical,
    highest_risk_engine,
    state_counts
  } = data;

  const healthyCount = state_counts.Healthy ?? 0;
  const degradingCount = state_counts.Degrading ?? 0;
  const criticalCount = state_counts.Critical ?? 0;

  const total = healthyCount + degradingCount + criticalCount || 1; // Prevent division by zero
  const healthyPct = (healthyCount / total) * 100;
  const degradingPct = (degradingCount / total) * 100;
  const criticalPct = (criticalCount / total) * 100;

  return (
    <div className="flex flex-col h-full justify-between gap-6">
      
      {/* Horizontal Stacked Bar */}
      <div className="flex flex-col w-full gap-2">
        <div className="flex w-full h-4 rounded-full overflow-hidden border border-border">
          {healthyCount > 0 && (
            <div style={{ width: `${healthyPct}%`, backgroundColor: STATE_VAR.Healthy }} />
          )}
          {degradingCount > 0 && (
            <div style={{ width: `${degradingPct}%`, backgroundColor: STATE_VAR.Degrading }} />
          )}
          {criticalCount > 0 && (
            <div style={{ width: `${criticalPct}%`, backgroundColor: STATE_VAR.Critical }} />
          )}
        </div>
        <div className="text-xs text-muted font-mono tracking-widest text-center">
          <span className={STATE_TEXT.Healthy}>Healthy {healthyCount}</span>
          <span className="mx-2">&middot;</span>
          <span className={STATE_TEXT.Degrading}>Degrading {degradingCount}</span>
          <span className="mx-2">&middot;</span>
          <span className={STATE_TEXT.Critical}>Critical {criticalCount}</span>
        </div>
      </div>

      {/* Font-mono Readouts */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Fleet Size */}
        <div className="flex flex-col bg-panel2 p-3 rounded-lg border border-border">
          <span className="text-[10px] font-mono text-muted uppercase tracking-wider mb-1">Fleet Size</span>
          <div className="text-2xl font-mono font-bold text-text">
            {n_engines}
          </div>
        </div>

        {/* Mean RUL */}
        <div className="flex flex-col bg-panel2 p-3 rounded-lg border border-border">
          <span className="text-[10px] font-mono text-muted uppercase tracking-wider mb-1">Mean RUL</span>
          <div className="text-2xl font-mono font-bold text-text">
            {mean_rul.toFixed(1)}
          </div>
        </div>

        {/* Median RUL */}
        <div className="flex flex-col bg-panel2 p-3 rounded-lg border border-border">
          <span className="text-[10px] font-mono text-muted uppercase tracking-wider mb-1">Median RUL</span>
          <div className="text-2xl font-mono font-bold text-text">
            {median_rul.toFixed(1)}
          </div>
        </div>

        {/* N Critical */}
        <div className="flex flex-col bg-panel2 p-3 rounded-lg border border-border">
          <span className="text-[10px] font-mono text-muted uppercase tracking-wider mb-1">N Critical</span>
          <div className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-full bg-critical"></div>
            <div className={`text-2xl font-mono font-bold ${STATE_TEXT.Critical}`}>
              {n_critical}
            </div>
          </div>
        </div>
      </div>

      {/* Action Button */}
      <div className="text-sm text-text">
        <span className="text-muted">Highest-risk engine:</span>{' '}
        <button 
          onClick={() => onSelectEngine(highest_risk_engine)}
          className={`font-mono font-bold ${STATE_TEXT.Critical} hover:underline underline-offset-4 focus:outline-none focus:ring-2 focus:ring-accent rounded-sm px-1`}
        >
          #{highest_risk_engine}
        </button>
      </div>

    </div>
  );
}
