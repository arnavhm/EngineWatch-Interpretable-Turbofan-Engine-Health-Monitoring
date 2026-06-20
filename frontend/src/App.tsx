import { useState } from 'react';
import TopBar from './components/TopBar';
import Selector from './components/Selector';
import MetricStrip from './components/MetricStrip';
import RiskDial from './components/RiskDial';
import FleetSummary from './components/FleetSummary';
import TopRiskTable from './components/TopRiskTable';
import TrajectoryPanel from './components/TrajectoryPanel';
import SensorPanel from './components/SensorPanel';
import AnomalyScatter from './components/AnomalyScatter';
import CsvUpload from './components/CsvUpload';
import EngineHealthMap from './components/EngineHealthMap';
import { usePredict } from './hooks/usePredict';

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
  const { data: engineData, loading: engineLoading, error: engineError } = usePredict(selectedEngine, selectedDataset);

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

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 pb-12">
          <Panel title="Engine Status" className="md:col-span-2 lg:col-span-3">
            <MetricStrip data={engineData} loading={engineLoading} error={engineError} />
          </Panel>
          <Panel title="Risk Dial" className="col-span-1">
            <RiskDial data={engineData} loading={engineLoading} />
          </Panel>
          <Panel title="Engine Health Map" className="md:col-span-1 lg:col-span-2">
            <EngineHealthMap engineId={selectedEngine} datasetId={selectedDataset} />
          </Panel>
          <Panel title="Fleet Summary" className="md:col-span-1 lg:col-span-2">
            <FleetSummary datasetId={selectedDataset} onSelectEngine={setSelectedEngine} />
          </Panel>
          <Panel title="Top Risk" className="md:col-span-2 lg:col-span-3">
            <TopRiskTable datasetId={selectedDataset} selectedEngine={selectedEngine} onSelectEngine={setSelectedEngine} />
          </Panel>
          <Panel title="HEALTH TRAJECTORY" className="md:col-span-2 lg:col-span-3">
            <TrajectoryPanel engineId={selectedEngine} datasetId={selectedDataset} />
          </Panel>
          <Panel title="SENSORS" className="md:col-span-2 lg:col-span-3">
            <SensorPanel engineId={selectedEngine} datasetId={selectedDataset} />
          </Panel>
          <Panel title="FLEET ANOMALY" className="md:col-span-2 lg:col-span-3">
            <AnomalyScatter datasetId={selectedDataset} />
          </Panel>
          <Panel title="CSV BATCH PREDICT" className="md:col-span-2 lg:col-span-3">
            <CsvUpload />
          </Panel>
        </div>
      </main>
    </div>
  );
}

export default App;
