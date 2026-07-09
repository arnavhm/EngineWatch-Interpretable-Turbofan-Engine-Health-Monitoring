import { useState, lazy, Suspense } from 'react';
import EngineStatusVerdict from './EngineStatusVerdict';
import AgentDiagnostic from './AgentDiagnostic';
import EngineHealthMap from './EngineHealthMap';
import StatStrip from './StatStrip';
import PanelState from './PanelState';
import { usePredict } from '../hooks/usePredict';

const TrajectoryPanel = lazy(() => import('./TrajectoryPanel'));
const VelocityPanel = lazy(() => import('./VelocityPanel'));
const VariabilityPanel = lazy(() => import('./VariabilityPanel'));
const SensorAccordion = lazy(() => import('./SensorAccordion'));

interface EngineDrillDownProps {
  engineId: number;
  datasetId: string;
  onBack: () => void;
}

export default function EngineDrillDown({ engineId, datasetId, onBack }: EngineDrillDownProps) {
  const [devMode, setDevMode] = useState(false);
  const { data: engineData, loading: engineLoading } = usePredict(engineId, datasetId);

  return (
    <div className="h-screen grid grid-cols-[35%_65%] overflow-hidden">
      <div className="h-screen overflow-y-auto border-r border-border p-6 flex flex-col gap-6">
        <button onClick={onBack} className="text-sm text-muted hover:text-text w-fit flex items-center gap-1">
          ← Back to Fleet Command
        </button>
        <EngineStatusVerdict engineId={engineId} datasetId={datasetId} />
        <AgentDiagnostic engineId={engineId} datasetId={datasetId} />
      </div>

      <div className="h-screen overflow-y-auto p-6 flex flex-col gap-6">
        <PanelState loading={engineLoading || !engineData} error={null}>
          {engineData && <StatStrip data={engineData} />}
        </PanelState>

        <EngineHealthMap engineId={engineId} datasetId={datasetId} />

        <button
          onClick={() => setDevMode(!devMode)}
          className="text-xs font-mono uppercase tracking-widest text-muted hover:text-text border border-border rounded-lg px-3 py-2 w-fit"
        >
          {devMode ? 'Hide' : 'View'} Raw Telemetry / Developer Mode
        </button>

        {devMode && (
          <div className="flex flex-col gap-6">
            <Suspense fallback={<PanelState loading={true}><div /></PanelState>}><TrajectoryPanel engineId={engineId} datasetId={datasetId} /></Suspense>
            <Suspense fallback={<PanelState loading={true}><div /></PanelState>}><VelocityPanel engineId={engineId} datasetId={datasetId} /></Suspense>
            <Suspense fallback={<PanelState loading={true}><div /></PanelState>}><VariabilityPanel engineId={engineId} datasetId={datasetId} /></Suspense>
            <Suspense fallback={<PanelState loading={true}><div /></PanelState>}><SensorAccordion engineId={engineId} datasetId={datasetId} /></Suspense>
          </div>
        )}
      </div>
    </div>
  );
}
