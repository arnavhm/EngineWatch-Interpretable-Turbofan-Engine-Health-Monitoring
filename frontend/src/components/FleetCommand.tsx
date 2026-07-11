import { lazy, Suspense } from 'react';
import DatasetSelector from './DatasetSelector';
import EngineSelector from './EngineSelector';
import FleetCompareTable from './FleetCompareTable';
import FleetSummary from './FleetSummary';
import TopRiskTable from './TopRiskTable';
import CsvUpload from './CsvUpload';
import PanelState from './PanelState';
import { useFleetAnalytics } from '../hooks/useFleetAnalytics';
import { useAnomaly } from '../hooks/useAnomaly';

const RiskHistogramChart = lazy(() => import('./RiskHistogramChart'));
const FleetTrendChart = lazy(() => import('./FleetTrendChart'));
const AnomalyScatter = lazy(() => import('./AnomalyScatter'));

interface FleetCommandProps {
  selectedDataset: string;
  setSelectedDataset: (id: string) => void;
  onSelectEngine: (id: number) => void;
}

function Panel({ title, children, className = '' }: { title: string; children?: React.ReactNode; className?: string }) {
  return (
    <div className={`bg-panel border border-border rounded-xl p-4 flex flex-col shadow-sm ${className}`}>
      <h2 className="text-sm font-bold text-text mb-4 uppercase tracking-wide">{title}</h2>
      <div className="flex-1 flex flex-col">{children}</div>
    </div>
  );
}

export default function FleetCommand({ selectedDataset, setSelectedDataset, onSelectEngine }: FleetCommandProps) {
  const { data: fleetData, loading: fleetLoading, error: fleetError } = useFleetAnalytics(selectedDataset);
  const { data: anomalyData, loading: anomalyLoading, error: anomalyError } = useAnomaly(selectedDataset);

  return (
    <main className="w-full max-w-[1180px] mx-auto px-4 sm:px-6">
      <div className="py-6 mb-2 flex flex-col sm:flex-row gap-4">
        <DatasetSelector selectedDataset={selectedDataset} setSelectedDataset={setSelectedDataset} />
        <EngineSelector
          points={anomalyData}
          loading={anomalyLoading}
          error={anomalyError}
          onSelectEngine={onSelectEngine}
        />
      </div>

      <h2 className="text-xl font-bold text-text mb-4 uppercase tracking-wide">Fleet Command</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-12">
        <Panel title="Fleet Compare" className="md:col-span-2 lg:col-span-3">
          <FleetCompareTable />
        </Panel>
        <Panel title="Risk Distribution" className="md:col-span-1 lg:col-span-2">
          <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
            <RiskHistogramChart data={fleetData} loading={fleetLoading} error={fleetError} />
          </Suspense>
        </Panel>
        <Panel title="Fleet State Counts" className="md:col-span-1 lg:col-span-1">
          <FleetSummary datasetId={selectedDataset} onSelectEngine={onSelectEngine} />
        </Panel>
        <Panel title="Risk Trend by Life %" className="md:col-span-2 lg:col-span-3">
          <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
            <FleetTrendChart data={fleetData} loading={fleetLoading} error={fleetError} />
          </Suspense>
        </Panel>
        <Panel title="Top Risk" className="md:col-span-2 lg:col-span-3">
          <TopRiskTable datasetId={selectedDataset} onSelectEngine={onSelectEngine} />
        </Panel>
        <Panel title="Fleet Anomaly" className="md:col-span-2 lg:col-span-3">
          <Suspense fallback={<PanelState loading={true}><div /></PanelState>}>
            <AnomalyScatter
              data={anomalyData}
              loading={anomalyLoading}
              error={anomalyError}
              onSelectEngine={onSelectEngine}
            />
          </Suspense>
        </Panel>
        <Panel title="CSV Batch Predict" className="md:col-span-2 lg:col-span-3">
          <CsvUpload />
        </Panel>
      </div>
    </main>
  );
}
