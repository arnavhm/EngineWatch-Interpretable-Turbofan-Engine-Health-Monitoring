import { useState, lazy, Suspense } from 'react';
import TopBar from './components/TopBar';
import Selector from './components/Selector';
import RiskDial from './components/RiskDial';
import FleetSummary from './components/FleetSummary';
import TopRiskTable from './components/TopRiskTable';
import CsvUpload from './components/CsvUpload';
import EngineHealthMap from './components/EngineHealthMap';
import EngineStatusVerdict from './components/EngineStatusVerdict';
import NarrationChat from './components/NarrationChat';
import { usePredict } from './hooks/usePredict';
import { useFleetAnalytics } from './hooks/useFleetAnalytics';
import FleetCompareTable from './components/FleetCompareTable';
import PanelState from './components/PanelState';

const RiskHistogramChart = lazy(() => import('./components/RiskHistogramChart'));
const FleetTrendChart = lazy(() => import('./components/FleetTrendChart'));
const TrajectoryPanel = lazy(() => import('./components/TrajectoryPanel'));
const VelocityPanel = lazy(() => import('./components/VelocityPanel'));
const VariabilityPanel = lazy(() => import('./components/VariabilityPanel'));
const SensorAccordion = lazy(() => import('./components/SensorAccordion'));
const AnomalyScatter = lazy(() => import('./components/AnomalyScatter'));

function Panel({ title, children, className = '' }: { title: string, children?: React.ReactNode, className?: string }) {
  return (
    <div className={`bg-panel border border-border rounded-xl p-4 flex flex-col shadow-sm ${className}`}>
      <h2 className="text-sm font-bold text-text mb-4 uppercase tracking-wide">{title}</h2>
      {children ? (
        <div className="flex-1 flex flex-col">
          {children}
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center text-muted text-sm font-mono border border-dashed border-border rounded-lg bg-panel2 bg-opacity-50 min-h-[150px]">
          Coming in Phase 2
        </div>
      )}
    </div>
  );
}

function App() {
  const [selectedDataset, setSelectedDataset] = useState<string>("FD001");
  const [selectedEngine, setSelectedEngine] = useState<number>(34);
  const { data: engineData, loading: engineLoading } = usePredict(selectedEngine, selectedDataset);
  const { data: fleetData, loading: fleetLoading, error: fleetError } = useFleetAnalytics(selectedDataset);

  return (
    <div className="min-h-screen bg-bg text-text font-sans flex flex-col overflow-x-hidden">
      <TopBar />
      
      <main className="flex-1 w-full max-w-[1180px] mx-auto px-4 sm:px-6">
        <div className="py-6 mb-2">
          <Selector 
            selectedDataset={selectedDataset}
            setSelectedDataset={setSelectedDataset}
            selectedEngine={selectedEngine}
            setSelectedEngine={setSelectedEngine}
          />
        </div>

        {/* Fleet Overview Section */}
        <div className="mb-10">
          <h2 className="text-xl font-bold text-text mb-4 uppercase tracking-wide">Fleet Overview</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Panel title="Fleet Compare" className="md:col-span-2 lg:col-span-3">
              <FleetCompareTable />
            </Panel>

            <Panel title="Risk Distribution" className="md:col-span-1 lg:col-span-2">
              <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
                <RiskHistogramChart data={fleetData} loading={fleetLoading} error={fleetError} />
              </Suspense>
            </Panel>

            <Panel title="Fleet State Counts" className="md:col-span-1 lg:col-span-1">
              <FleetSummary datasetId={selectedDataset} onSelectEngine={setSelectedEngine} />
            </Panel>

            <Panel title="Risk Trend by Life %" className="md:col-span-2 lg:col-span-3">
              <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
                <FleetTrendChart data={fleetData} loading={fleetLoading} error={fleetError} />
              </Suspense>
            </Panel>
          </div>
        </div>

        {/* Engine Diagnostics Section */}
        <div>
          <h2 className="text-xl font-bold text-text mb-4 uppercase tracking-wide">Engine Diagnostics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-12">
            <div className="md:col-span-2 lg:col-span-3">
              <EngineStatusVerdict engineId={selectedEngine} datasetId={selectedDataset} />
            </div>

          <Panel title="Risk Dial" className="col-span-1">
            <RiskDial data={engineData} loading={engineLoading} />
          </Panel>
          <Panel title="Engine Health Map" className="md:col-span-1 lg:col-span-2">
            <EngineHealthMap engineId={selectedEngine} datasetId={selectedDataset} />
          </Panel>
          
          <Panel title="Agentic AI Diagnostic Assistant" className="md:col-span-1 lg:col-span-1 md:row-span-2 lg:row-span-1">
            <NarrationChat engineId={selectedEngine} datasetId={selectedDataset} />
          </Panel>

          <Panel title="Degradation Velocity" className="col-span-1 lg:col-span-1">
            <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
              <VelocityPanel engineId={selectedEngine} datasetId={selectedDataset} />
            </Suspense>
          </Panel>
          <Panel title="Health Variability" className="col-span-1 lg:col-span-1">
            <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
              <VariabilityPanel engineId={selectedEngine} datasetId={selectedDataset} />
            </Suspense>
          </Panel>

          <Panel title="HEALTH TRAJECTORY" className="md:col-span-2 lg:col-span-3">
            <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
              <TrajectoryPanel engineId={selectedEngine} datasetId={selectedDataset} />
            </Suspense>
          </Panel>

          <Panel title="SENSORS" className="md:col-span-2 lg:col-span-3">
            <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
              <SensorAccordion engineId={selectedEngine} datasetId={selectedDataset} />
            </Suspense>
          </Panel>
          
          <Panel title="Top Risk" className="md:col-span-2 lg:col-span-3">
            <TopRiskTable datasetId={selectedDataset} selectedEngine={selectedEngine} onSelectEngine={setSelectedEngine} />
          </Panel>

          <Panel title="FLEET ANOMALY" className="md:col-span-2 lg:col-span-3">
            <Suspense fallback={<PanelState loading={true}><div/></PanelState>}>
              <AnomalyScatter datasetId={selectedDataset} />
            </Suspense>
          </Panel>
          
          <Panel title="CSV BATCH PREDICT" className="md:col-span-2 lg:col-span-3">
            <CsvUpload />
          </Panel>
        </div>
        </div>
      </main>
    </div>
  );
}

export default App;
