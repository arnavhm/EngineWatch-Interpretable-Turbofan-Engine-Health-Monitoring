import { useState, useEffect } from 'react';
import TopBar from './components/TopBar';
import FleetCommand from './components/FleetCommand';
import EngineDrillDown from './components/EngineDrillDown';

function parseRoute(): { view: 'fleet' | 'engine'; engineId: number | null } {
  const match = window.location.pathname.match(/^\/engine\/(\d+)$/);
  return match ? { view: 'engine', engineId: Number(match[1]) } : { view: 'fleet', engineId: null };
}

function App() {
  const [route, setRoute] = useState(parseRoute());
  const [selectedDataset, setSelectedDataset] = useState<string>('FD001');

  useEffect(() => {
    const onPopState = () => setRoute(parseRoute());
    window.addEventListener('popstate', onPopState);
    return () => window.removeEventListener('popstate', onPopState);
  }, []);

  const goToEngine = (id: number) => {
    window.history.pushState(null, '', `/engine/${id}`);
    setRoute({ view: 'engine', engineId: id });
  };

  const goToFleet = () => {
    window.history.pushState(null, '', '/');
    setRoute({ view: 'fleet', engineId: null });
  };

  return route.view === 'engine' && route.engineId !== null ? (
    <div className="h-screen flex flex-col bg-bg text-text font-sans overflow-hidden">
      <TopBar />
      <div className="flex-1 min-h-0">
        <EngineDrillDown engineId={route.engineId} datasetId={selectedDataset} onBack={goToFleet} />
      </div>
    </div>
  ) : (
    <div className="min-h-screen bg-bg text-text font-sans overflow-x-hidden">
      <TopBar />
      <FleetCommand
        selectedDataset={selectedDataset}
        setSelectedDataset={setSelectedDataset}
        onSelectEngine={goToEngine}
      />
    </div>
  );
}

export default App;
