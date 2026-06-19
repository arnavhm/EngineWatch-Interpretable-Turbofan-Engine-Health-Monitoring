import { useEffect, useState } from 'react';
import { getFleetSummary } from '../api';
import type { FleetSummary } from '../types';

const DATASETS = [
  { id: 'FD001', label: 'FD001 — Baseline · 1 condition, 1 fault (HPC)' },
  { id: 'FD002', label: 'FD002 — Multi-regime · 6 conditions, 1 fault (HPC)' },
  { id: 'FD003', label: 'FD003 — Dual-fault · 1 condition, 2 faults (HPC + Fan)' },
  { id: 'FD004', label: 'FD004 — Complex · 6 conditions, 2 faults (HPC + Fan)' },
];

interface SelectorProps {
  selectedDataset: string;
  setSelectedDataset: (id: string) => void;
  selectedEngine: number;
  setSelectedEngine: (id: number) => void;
}

export default function Selector({
  selectedDataset,
  setSelectedDataset,
  selectedEngine,
  setSelectedEngine,
}: SelectorProps) {
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialMount, setIsInitialMount] = useState(true);

  useEffect(() => {
    let mounted = true;
    setIsLoading(true);
    getFleetSummary(selectedDataset)
      .then((data) => {
        if (mounted) {
          setSummary(data);
        }
      })
      .catch((err) => console.error("Failed to load fleet summary:", err))
      .finally(() => {
        if (mounted) setIsLoading(false);
      });
    return () => { mounted = false; };
  }, [selectedDataset]);

  useEffect(() => {
    if (summary) {
      if (isInitialMount) {
        setIsInitialMount(false);
        if (selectedEngine > summary.n_engines) {
          setSelectedEngine(1);
        }
      } else {
        setSelectedEngine(summary.highest_risk_engine);
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [summary]); // We ONLY want to trigger this when summary updates (i.e. dataset changes).

  const nEngines = summary?.n_engines || 0;
  const engineOptions = Array.from({ length: nEngines }, (_, i) => i + 1);

  return (
    <div className="flex flex-col sm:flex-row gap-6 items-center w-full">
      <div className="flex flex-col w-full sm:flex-1">
        <label className="text-[10px] text-muted mb-1.5 font-mono uppercase tracking-widest font-semibold">Dataset</label>
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="bg-panel2 border border-border rounded-md text-text font-mono px-3 py-2 outline-none focus:border-accent w-full transition-colors shadow-sm"
        >
          {DATASETS.map((ds) => (
            <option key={ds.id} value={ds.id}>
              {ds.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-col w-full sm:w-48">
        <label className="text-[10px] text-muted mb-1.5 font-mono uppercase tracking-widest font-semibold">Engine Unit</label>
        <select
          value={selectedEngine}
          onChange={(e) => setSelectedEngine(Number(e.target.value))}
          disabled={isLoading || nEngines === 0}
          className="bg-panel2 border border-border rounded-md text-text font-mono px-3 py-2 outline-none focus:border-accent w-full disabled:opacity-50 transition-colors shadow-sm"
        >
          {engineOptions.length > 0 ? (
            engineOptions.map((num) => (
              <option key={num} value={num}>
                {num}
              </option>
            ))
          ) : (
            <option value={selectedEngine}>{selectedEngine}</option>
          )}
        </select>
      </div>
    </div>
  );
}
