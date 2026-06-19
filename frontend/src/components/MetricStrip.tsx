import { usePredict } from '../hooks/usePredict';
import { STATE_TEXT, STATE_VAR } from '../stateColors';
import PanelState from './PanelState';

interface MetricStripProps {
  engineId: number;
  datasetId: string;
}

export default function MetricStrip({ engineId, datasetId }: MetricStripProps) {
  const { data, loading, error } = usePredict(engineId, datasetId);

  return (
    <PanelState loading={loading || !data} error={error}>
      {data && (() => {
        const { rul_cycles, ci_lower, ci_upper, risk_score, health_index, risk_state } = data;
        const colorClass = STATE_TEXT[risk_state];
        const colorVar = STATE_VAR[risk_state];

        return (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full h-full">
            {/* RUL */}
            <div className="flex flex-col justify-center bg-panel2 p-4 rounded-lg border border-border">
              <span className="text-xs font-mono text-muted uppercase tracking-wider mb-1">RUL</span>
              <div className={`text-3xl font-mono font-bold ${colorClass}`}>
                {rul_cycles.toFixed(1)}
              </div>
              <span className="text-xs text-faint mt-1 font-mono">
                95% CI {ci_lower.toFixed(1)}–{ci_upper.toFixed(1)}
              </span>
            </div>

            {/* Risk Score */}
            <div className="flex flex-col justify-center bg-panel2 p-4 rounded-lg border border-border">
              <span className="text-xs font-mono text-muted uppercase tracking-wider mb-1">Risk Score</span>
              <div className={`text-3xl font-mono font-bold ${colorClass}`}>
                {risk_score.toFixed(3)}
              </div>
              <span className="text-xs text-faint mt-1 font-mono">
                0–1 scale
              </span>
            </div>

            {/* Health Index */}
            <div className="flex flex-col justify-center bg-panel2 p-4 rounded-lg border border-border">
              <span className="text-xs font-mono text-muted uppercase tracking-wider mb-1">Health Index</span>
              <div className="text-3xl font-mono font-bold text-text">
                {health_index.toFixed(3)}
              </div>
              <span className="text-xs text-faint mt-1 font-mono">
                PC1 &middot; 64.3% var
              </span>
            </div>

            {/* State Chip */}
            <div className="flex flex-col justify-center items-center bg-panel2 p-4 rounded-lg border border-border">
              <div 
                className="flex items-center gap-2 px-4 py-2 rounded-full border border-solid"
                style={{ 
                  backgroundColor: `color-mix(in srgb, ${colorVar} 12%, transparent)`,
                  borderColor: `color-mix(in srgb, ${colorVar} 35%, transparent)`
                }}
              >
                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: colorVar }}></div>
                <span className="text-sm font-mono font-bold tracking-widest" style={{ color: colorVar }}>
                  {risk_state.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
        );
      })()}
    </PanelState>
  );
}
