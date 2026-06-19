import { useEffect, useState } from 'react';
import { checkHealth } from '../api';

export default function TopBar() {
  const [status, setStatus] = useState<'LOADING' | 'ONLINE' | 'OFFLINE'>('LOADING');

  useEffect(() => {
    let mounted = true;
    checkHealth().then((isHealthy) => {
      if (mounted) {
        setStatus(isHealthy ? 'ONLINE' : 'OFFLINE');
      }
    });
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <header className="flex items-center justify-between px-6 py-4 border-b border-border bg-bg">
      <div className="flex items-center space-x-4">
        {/* Brand Mark */}
        <div className="flex items-center justify-center w-8 h-8 border-2 border-healthy rounded-md">
          <span className="text-healthy font-bold text-lg leading-none">E</span>
        </div>
        {/* Title & Subtitle */}
        <div className="flex flex-col">
          <h1 className="text-text font-bold text-lg leading-tight tracking-tight">EngineWatch</h1>
          <span className="text-muted text-xs">Interpretable Turbofan Prognostics &middot; C-MAPSS</span>
        </div>
      </div>

      {/* API Status */}
      <div className="flex items-center space-x-2 bg-panel px-3 py-1.5 rounded-full border border-border">
        <div 
          className={`w-2 h-2 rounded-full ${status === 'ONLINE' ? 'bg-healthy' : status === 'OFFLINE' ? 'bg-critical' : 'bg-muted'}`}
        ></div>
        <span className="font-mono text-xs text-text">{status}</span>
      </div>
    </header>
  );
}
