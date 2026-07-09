const DATASETS = [
  { id: 'FD001', label: 'FD001 — Baseline · 1 condition, 1 fault (HPC)' },
  { id: 'FD002', label: 'FD002 — Multi-regime · 6 conditions, 1 fault (HPC)' },
  { id: 'FD003', label: 'FD003 — Dual-fault · 1 condition, 2 faults (HPC + Fan)' },
  { id: 'FD004', label: 'FD004 — Complex · 6 conditions, 2 faults (HPC + Fan)' },
];

interface DatasetSelectorProps {
  selectedDataset: string;
  setSelectedDataset: (id: string) => void;
}

export default function DatasetSelector({
  selectedDataset,
  setSelectedDataset,
}: DatasetSelectorProps) {
  return (
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
  );
}
