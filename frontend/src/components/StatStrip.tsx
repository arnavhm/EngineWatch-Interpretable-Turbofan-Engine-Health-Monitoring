import { STATE_TEXT } from '../stateColors';
import type { PredictResponse } from '../types';

export default function StatStrip({ data }: { data: PredictResponse }) {
  const stateColorClass = STATE_TEXT[data.risk_state];
  const isCritical = data.risk_state === 'Critical';
  const isPastThreshold = data.health_index < 0.30;

  return (
    <div className="grid grid-cols-3 gap-4">
      <div className="bg-panel2 border border-border rounded-lg px-4 py-3">
        <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted">
          CONDITION
        </div>
        <div className={`text-xl font-semibold ${stateColorClass}`}>{data.risk_state}</div>
        <div className="text-xs font-mono text-faint">RISK {data.risk_score.toFixed(2)}</div>
      </div>

      <div className="bg-panel2 border border-border rounded-lg px-4 py-3">
        <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted">
          {isCritical ? 'FAILS IN' : 'USEFUL LIFE LEFT'}
        </div>
        <div className={`text-2xl font-semibold ${isCritical ? stateColorClass : 'text-text'}`}>
          ~{Math.round(data.rul_cycles)} flights
        </div>
        <div className="text-xs font-mono text-faint">
          RUL {data.rul_cycles.toFixed(1)} CYCLES · CI {data.ci_lower.toFixed(1)}–
          {data.ci_upper.toFixed(1)}
        </div>
      </div>

      <div className="bg-panel2 border border-border rounded-lg px-4 py-3">
        <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted">
          HEALTH INDEX
        </div>
        <div
          className={`text-base font-semibold ${isPastThreshold ? 'text-critical' : 'text-text'}`}
        >
          {isPastThreshold ? 'Past critical threshold' : 'Above critical threshold'}
        </div>
        <div className="text-xs font-mono text-faint">
          HI {data.health_index.toFixed(2)} · THRESHOLD 0.30
        </div>
      </div>
    </div>
  );
}
