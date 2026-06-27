
import { usePredict } from '../hooks/usePredict';
import { useContributions } from '../hooks/useContributions';
import { useTrajectory } from '../hooks/useTrajectory';

interface EngineStatusVerdictProps {
  engineId: number;
  datasetId: string;
}

export default function EngineStatusVerdict({ engineId, datasetId }: EngineStatusVerdictProps) {
  const { data: pData, loading: pLoading } = usePredict(engineId, datasetId);
  const { data: cData, loading: cLoading } = useContributions(engineId, datasetId);
  const { data: tData, loading: tLoading } = useTrajectory(engineId, datasetId);

  if (pLoading || cLoading || tLoading) {
    return <div className="animate-pulse bg-panel2 h-16 rounded border border-border"></div>;
  }

  if (!pData || !cData || !tData) {
    return null;
  }

  const {
    risk_state,
    rul_cycles,
    ci_lower,
    ci_upper,
    health_index,
    rmse
  } = pData;

  const {
    dominant_module,
    dominant_driver_text
  } = cData;

  const lastVelocity = tData.velocity[tData.velocity.length - 1] ?? 0;
  
  const isPast = health_index < 0.30;
  const isDropping = lastVelocity < 0;

  const borderColors: Record<string, string> = {
    Critical: '#E0533A',
    Degrading: '#E0A93A',
    Healthy: '#3ECF8E',
  };

  const borderColor = borderColors[risk_state] || '#3ECF8E';

  return (
    <div 
      className="bg-panel p-4 rounded border border-border shadow-sm text-sm"
      style={{ borderLeft: `4px solid ${borderColor}` }}
    >
      <span className="font-bold" style={{ color: borderColor }}>{risk_state}</span> · fails in ~{Math.round(rul_cycles)} cycles (RUL {rul_cycles.toFixed(2)}, CI {ci_lower.toFixed(1)}–{ci_upper.toFixed(1)}).{' '}
      Health index has fallen to {health_index.toFixed(2)}, {isPast ? 'past' : 'approaching'} the 0.30 critical threshold{isDropping ? ', and is still dropping' : ' and holding'}.{' '}
      Driven by the {dominant_module.toUpperCase()} — {dominant_driver_text}.{' '}
      Model: Monotonic HistGradientBoostingRegressor · RMSE {rmse.toFixed(2)}.
    </div>
  );
}
