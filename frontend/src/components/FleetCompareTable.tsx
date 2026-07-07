import { useFleetCompare } from '../hooks/useFleetCompare';
import PanelState from './PanelState';
import { STATE_TEXT } from '../stateColors';

export default function FleetCompareTable() {
  const { data, loading, error } = useFleetCompare();

  return (
    <PanelState loading={loading || !data} error={error}>
      {data && (
        <div className="w-full overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-muted uppercase bg-panel2 border-b border-border">
              <tr>
                <th className="px-4 py-3">Dataset</th>
                <th className="px-4 py-3">Fleet Size</th>
                <th className="px-4 py-3">State Counts</th>
                <th className="px-4 py-3">N Critical</th>
                <th className="px-4 py-3">Mean RUL</th>
                <th className="px-4 py-3">Median RUL</th>
              </tr>
            </thead>
            <tbody>
              {data.map((row) => {
                const h = row.state_counts.Healthy ?? 0;
                const d = row.state_counts.Degrading ?? 0;
                const c = row.state_counts.Critical ?? 0;
                
                return (
                  <tr key={row.dataset_id} className="border-b border-border last:border-0 hover:bg-panel2/50 transition-colors">
                    <td className="px-4 py-3 font-mono font-bold">{row.dataset_id}</td>
                    <td className="px-4 py-3 font-mono">{row.fleet_size}</td>
                    <td className="px-4 py-3 font-mono text-xs">
                      <span className={STATE_TEXT.Healthy}>{h}H</span> /{' '}
                      <span className={STATE_TEXT.Degrading}>{d}D</span> /{' '}
                      <span className={STATE_TEXT.Critical}>{c}C</span>
                    </td>
                    <td className="px-4 py-3 font-mono text-critical">{row.n_critical}</td>
                    <td className="px-4 py-3 font-mono">{row.mean_rul.toFixed(1)}</td>
                    <td className="px-4 py-3 font-mono">{row.median_rul.toFixed(1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </PanelState>
  );
}
