
import { useContributions } from '../hooks/useContributions';
import { useTrajectory } from '../hooks/useTrajectory';
import { STATE_TEXT } from '../stateColors';
import type { PredictResponse } from '../types';

interface EngineStatusVerdictProps {
  data: PredictResponse;
  engineId: number;
  datasetId: string;
}

const BORDER_COLORS: Record<string, string> = {
  Critical: 'border-l-critical',
  Degrading: 'border-l-degrading',
  Healthy: 'border-l-healthy',
};

export default function EngineStatusVerdict({ data, engineId, datasetId }: EngineStatusVerdictProps) {
  const { data: cData, loading: cLoading } = useContributions(engineId, datasetId);
  const { data: tData, loading: tLoading } = useTrajectory(engineId, datasetId);

  if (cLoading || tLoading) {
    return <div className="animate-pulse bg-panel2 h-32 rounded-lg border border-border"></div>;
  }

  if (!cData || !tData) {
    return null;
  }

  const {
    risk_state,
    rul_cycles,
    ci_lower,
    ci_upper,
    health_index,
    risk_score,
    model_name,
    rmse,
  } = data;

  const { dominant_driver_text } = cData;

  const lastVelocity = tData.velocity[tData.velocity.length - 1] ?? 0;

  const isPast = health_index < 0.30;
  const isDropping = lastVelocity < 0;

  const stateColorClass = STATE_TEXT[risk_state];
  const borderColorClass = BORDER_COLORS[risk_state];

  const rulRounded = Math.round(rul_cycles);

  const headline = (() => {
    if (risk_state === 'Critical') {
      return (
        <>
          Engine {engineId} is{' '}
          <span className={`font-bold uppercase ${stateColorClass}`}>CRITICAL</span> — likely to
          fail in about {rulRounded} flights.
        </>
      );
    }
    if (risk_state === 'Degrading') {
      return (
        <>
          Engine {engineId} is{' '}
          <span className={`font-bold uppercase ${stateColorClass}`}>DEGRADING</span> — about{' '}
          {rulRounded} flights of useful life left.
        </>
      );
    }
    return (
      <>
        Engine {engineId} is{' '}
        <span className={`font-bold uppercase ${stateColorClass}`}>HEALTHY</span> — about{' '}
        {rulRounded} flights of useful life left.
      </>
    );
  })();

  return (
    <div className={`bg-panel border border-border rounded-lg p-5 border-l-4 ${borderColorClass}`}>
      <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted">
        ENGINE {engineId} · {datasetId.toUpperCase()}
      </div>

      <div className="text-2xl font-semibold leading-snug text-text mt-2">{headline}</div>

      <div className="text-xs font-mono text-faint mt-2">
        RUL {rul_cycles.toFixed(1)} · RISK {risk_score.toFixed(2)} · CI {ci_lower.toFixed(1)}–
        {ci_upper.toFixed(1)} · HEALTH INDEX {health_index.toFixed(2)}
      </div>

      <div className="text-sm text-muted leading-relaxed mt-3">
        Health index has fallen to {health_index.toFixed(2)},{' '}
        {isPast ? 'past' : 'approaching'} the 0.30 critical threshold
        {isDropping ? ', and is still dropping' : ' and holding'}. Driven by the{' '}
        {dominant_driver_text}.
      </div>

      <div className="text-[10px] font-mono text-faint mt-3">
        {model_name} · RMSE {rmse.toFixed(2)}
      </div>
    </div>
  );
}
