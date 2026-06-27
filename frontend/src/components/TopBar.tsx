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
        <div className="flex items-center justify-center w-9 h-9 rounded-md overflow-hidden bg-[#0a111a] border border-[#263040]">
          <img src="/logo.jpg" alt="EngineWatch Logo" className="w-full h-full object-cover scale-[1.2]" />
        </div>
        {/* Title & Subtitle */}
        <div className="flex flex-col">
          <h1 className="text-text font-bold text-lg leading-tight tracking-tight">EngineWatch</h1>
          <span className="text-muted text-xs">Interpretable Turbofan Prognostics &middot; C-MAPSS</span>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        {/* Creator Badge */}
        <a 
          href="https://github.com/arnavhm" 
          target="_blank" 
          rel="noopener noreferrer"
          className="hidden sm:flex items-center space-x-2 bg-panel hover:bg-panel2 transition-colors px-3 py-1.5 rounded-full border border-border"
        >
          <span className="text-xs text-muted">Built by</span>
          <span className="text-xs font-bold text-text">Arnav</span>
          <svg className="w-3 h-3 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" /></svg>
        </a>

        {/* API Status */}
        <div className="flex items-center space-x-2 bg-panel px-3 py-1.5 rounded-full border border-border">
          <div 
            className={`w-2 h-2 rounded-full ${status === 'ONLINE' ? 'bg-healthy' : status === 'OFFLINE' ? 'bg-critical' : 'bg-muted'}`}
          ></div>
          <span className="font-mono text-xs text-text">{status}</span>
        </div>
      </div>
    </header>
  );
}
