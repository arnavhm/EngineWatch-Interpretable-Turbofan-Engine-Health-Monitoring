import { STATE_TEXT } from '../stateColors';
import type { PredictResponse } from '../types';

export default function StatStrip({ data }: { data: PredictResponse }) {
  const colorClass = STATE_TEXT[data.risk_state];
  return (
    <div className="flex gap-4 flex-wrap">
      <div className="bg-panel2 border border-border rounded-lg px-3 py-2">
        <div className="text-[10px] text-muted uppercase tracking-wider">Risk score</div>
        <div className={`text-lg font-mono font-bold ${colorClass}`}>{data.risk_score.toFixed(2)}</div>
      </div>
      <div className="bg-panel2 border border-border rounded-lg px-3 py-2">
        <div className="text-[10px] text-muted uppercase tracking-wider">RUL (cycles)</div>
        <div className="text-lg font-mono font-bold text-text">{data.rul_cycles.toFixed(1)}</div>
      </div>
    </div>
  );
}
