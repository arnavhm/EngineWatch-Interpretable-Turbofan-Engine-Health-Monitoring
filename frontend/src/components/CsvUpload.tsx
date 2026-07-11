import { useState } from 'react';
import { predictCsv } from '../api';
import type { CsvPrediction } from '../types';

const STATE_COLOR: Record<string, string> = {
  Healthy: 'var(--color-healthy)',
  Degrading: 'var(--color-degrading)',
  Critical: 'var(--color-critical)',
};

export default function CsvUpload({ datasetId }: { datasetId: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CsvPrediction[] | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const data = await predictCsv(datasetId, file);
      setResults(data.predictions || []);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      {/* Upload controls */}
      <div className="flex items-center gap-3">
        <label htmlFor="csv-upload-input" className="flex-1 cursor-pointer">
          <div className={`flex items-center justify-center px-4 py-3 rounded-lg border border-dashed transition-colors ${
            file ? 'border-accent bg-accent/5 text-accent' : 'border-border text-muted hover:border-faint'
          }`}>
            <span className="text-sm font-mono">
              {file ? file.name : 'Choose CMAPSS sensor CSV…'}
            </span>
          </div>
          <input
            id="csv-upload-input"
            type="file"
            accept=".csv"
            className="hidden"
            onChange={e => setFile(e.target.files?.[0] || null)}
          />
        </label>
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className={`px-5 py-3 rounded-lg text-sm font-bold font-mono uppercase tracking-wide transition-colors cursor-pointer ${
            !file || loading
              ? 'bg-panel2 text-faint border border-border cursor-not-allowed'
              : 'bg-accent text-bg border border-accent hover:bg-accent/90'
          }`}
        >
          {loading ? 'Scoring…' : 'Score'}
        </button>
      </div>

      {/* Error state */}
      {error && (
        <div className="bg-panel2 bg-opacity-50 border border-dashed border-border rounded-lg px-4 py-3">
          <span className="text-muted text-sm border-l-2 border-critical pl-4">Error: {error}</span>
        </div>
      )}

      {/* Results table */}
      {results && results.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-panel2">
                <th className="px-4 py-2.5 text-left text-xs font-bold font-mono uppercase text-muted tracking-wide">Engine</th>
                <th className="px-4 py-2.5 text-left text-xs font-bold font-mono uppercase text-muted tracking-wide">Risk Score</th>
                <th className="px-4 py-2.5 text-left text-xs font-bold font-mono uppercase text-muted tracking-wide">RUL Cycles</th>
                <th className="px-4 py-2.5 text-left text-xs font-bold font-mono uppercase text-muted tracking-wide">State</th>
              </tr>
            </thead>
            <tbody>
              {results.map((row, i) => (
                <tr key={i} className="border-b border-border/50 hover:bg-panel2/50 transition-colors">
                  <td className="px-4 py-2.5 font-mono text-text">{row.engine_id}</td>
                  <td className="px-4 py-2.5 font-mono text-text">{typeof row.risk_score === 'number' ? row.risk_score.toFixed(3) : '—'}</td>
                  <td className="px-4 py-2.5 font-mono text-text">{typeof row.rul_cycles === 'number' ? Math.round(row.rul_cycles) : '—'}</td>
                  <td className="px-4 py-2.5 font-mono font-bold" style={{ color: STATE_COLOR[row.risk_state] || 'var(--color-text)' }}>
                    {row.risk_state}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {results && results.length === 0 && (
        <div className="text-muted text-sm font-mono text-center py-4">No predictions returned.</div>
      )}
    </div>
  );
}
