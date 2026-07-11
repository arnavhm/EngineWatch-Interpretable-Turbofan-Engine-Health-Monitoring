import { useState, lazy, Suspense } from 'react';
import EngineStatusVerdict from './EngineStatusVerdict';
import AgentDiagnostic from './AgentDiagnostic';
import EngineHealthMap from './EngineHealthMap';
import StatStrip from './StatStrip';
import PanelState from './PanelState';
import { usePredict } from '../hooks/usePredict';
import { useContributions } from '../hooks/useContributions';
import { useTrajectory } from '../hooks/useTrajectory';
import type { RiskState } from '../types';

const TrajectoryPanel = lazy(() => import('./TrajectoryPanel'));
const VelocityPanel = lazy(() => import('./VelocityPanel'));
const VariabilityPanel = lazy(() => import('./VariabilityPanel'));
const SensorAccordion = lazy(() => import('./SensorAccordion'));

interface EngineDrillDownProps {
  engineId: number;
  datasetId: string;
  onBack: () => void;
}

const ACTION_BORDER_COLORS: Record<RiskState, string> = {
  Critical: 'border-l-critical',
  Degrading: 'border-l-degrading',
  Healthy: 'border-l-healthy',
};

const NARRATION_QUESTION: Record<RiskState, string> = {
  Critical: 'Why is this engine critical?',
  Degrading: 'Why is this engine degrading?',
  Healthy: 'Why is this engine healthy?',
};

export default function EngineDrillDown({ engineId, datasetId, onBack }: EngineDrillDownProps) {
  const [devMode, setDevMode] = useState(false);
  const { data: engineData, loading: engineLoading } = usePredict(engineId, datasetId);
  const { data: contribData, loading: contribLoading, error: contribError } = useContributions(engineId, datasetId);
  const { data: trajData, loading: trajLoading, error: trajError } = useTrajectory(engineId, datasetId);

  if (engineLoading || !engineData) {
    return (
      <div className="h-full w-full flex items-center justify-center bg-bg">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-border border-t-muted"></div>
      </div>
    );
  }

  const N = Math.round(engineData.rul_cycles);

  const actionLine: Record<RiskState, string> = {
    Critical: `→ Schedule inspection within ~${N} flights.`,
    Degrading: `→ Plan maintenance at the next scheduled opportunity — about ${N} flights of margin.`,
    Healthy: '→ No action needed. Continue routine monitoring.',
  };

  const actionContext: Record<RiskState, string> = {
    Critical: `Failure is likely before ${N} more flights are completed.`,
    Degrading: 'Condition is worsening but has not reached the critical threshold.',
    Healthy: 'All health signals are within normal range.',
  };

  return (
    <div className="h-full grid grid-cols-[35%_65%] overflow-hidden">
      <div className="h-full overflow-y-auto border-r border-border p-6 flex flex-col gap-5">
        <button onClick={onBack} className="text-sm text-muted hover:text-text w-fit flex items-center gap-1">
          ← Back to Fleet Command
        </button>

        <EngineStatusVerdict
          data={engineData}
          engineId={engineId}
          datasetId={datasetId}
          contributions={contribData}
          contributionsLoading={contribLoading}
          trajectory={trajData}
          trajectoryLoading={trajLoading}
        />

        <div>
          <div className="text-sm font-semibold text-text border-b border-border pb-2">
            {NARRATION_QUESTION[engineData.risk_state]}
          </div>
          <div className="mt-3">
            <AgentDiagnostic engineId={engineId} datasetId={datasetId} />
          </div>
        </div>

        <div
          className={`mt-auto pt-5 bg-panel2 border border-border rounded-lg p-4 border-l-4 ${ACTION_BORDER_COLORS[engineData.risk_state]}`}
        >
          <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted">
            RECOMMENDED ACTION
          </div>
          <div className="text-base font-semibold text-text mt-1">
            {actionLine[engineData.risk_state]}
          </div>
          <div className="text-xs font-mono text-faint mt-1">
            {actionContext[engineData.risk_state]}
          </div>
        </div>
      </div>

      <div className="h-full overflow-y-auto p-6 flex flex-col gap-5">
        <StatStrip data={engineData} />

        <div className="flex items-center justify-between">
          <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted">
            ENGINE HEALTH MAP
          </div>
          <div className="text-xs text-muted">◉ Tap any engine section to inspect it</div>
        </div>

        <div className="flex-1">
          <EngineHealthMap contributions={contribData} loading={contribLoading} error={contribError} />
        </div>

        <button
          onClick={() => setDevMode(!devMode)}
          aria-pressed={devMode}
          className="text-xs font-mono uppercase tracking-widest text-muted hover:text-text border border-border rounded-lg px-3 py-2 w-fit"
        >
          {devMode ? 'Hide sensor details ▾' : 'Inspect sensor details ▸'}
        </button>

        {devMode && (
          <div className="flex flex-col gap-6">
            <div className="bg-panel border border-border rounded-lg p-4">
              <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted mb-3">
                HEALTH TRAJECTORY
              </div>
              <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
                <TrajectoryPanel data={trajData} loading={trajLoading} error={trajError} />
              </Suspense>
            </div>

            <div className="bg-panel border border-border rounded-lg p-4">
              <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted mb-3">
                DEGRADATION SPEED
              </div>
              <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
                <VelocityPanel data={trajData} loading={trajLoading} error={trajError} />
              </Suspense>
            </div>

            <div className="bg-panel border border-border rounded-lg p-4">
              <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted mb-3">
                SIGNAL STABILITY
              </div>
              <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
                <VariabilityPanel data={trajData} loading={trajLoading} error={trajError} />
              </Suspense>
            </div>

            <div className="bg-panel border border-border rounded-lg p-4">
              <div className="text-[11px] font-mono uppercase tracking-[0.18em] text-muted mb-3">
                SENSOR DETAILS
              </div>
              <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
                <SensorAccordion
                  engineId={engineId}
                  datasetId={datasetId}
                  dominantDriverText={contribData?.dominant_driver_text ?? null}
                />
              </Suspense>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
