import { useMemo } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import PanelState from './PanelState';
import type { AnomalyPoint } from '../types';

interface AnomalyScatterProps {
  data: AnomalyPoint[] | null;
  loading: boolean;
  error: string | null;
  onSelectEngine: (id: number) => void;
}

const STATE_COLOR: Record<string, string> = {
  Healthy: 'var(--color-healthy)',
  Degrading: 'var(--color-degrading)',
  Critical: 'var(--color-critical)',
};

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const d = payload[0].payload;
    return (
      <div className="bg-panel2 border border-border p-2 rounded shadow-md text-xs font-mono">
        <p className="text-muted mb-1">Engine {d.engine_id}</p>
        <p className="text-text">HI: <span className="text-accent font-bold">{d.health_index.toFixed(3)}</span></p>
        <p className="text-text">Vel: <span className="text-accent font-bold">{d.velocity.toFixed(4)}</span></p>
        <p className="text-text">State: <span style={{ color: STATE_COLOR[d.risk_state] || 'var(--color-muted)' }}>{d.risk_state}</span></p>
        {d.is_anomaly && <p className="text-critical font-bold mt-1">⚠ ANOMALY</p>}
      </div>
    );
  }
  return null;
};

export default function AnomalyScatter({ data, loading, error, onSelectEngine }: AnomalyScatterProps) {
  const handlePointClick = (point: any) => {
    const engineId = point?.payload?.engine_id ?? point?.engine_id;
    if (typeof engineId === 'number') onSelectEngine(engineId);
  };

  const { normal, anomalies } = useMemo(() => {
    if (!data) return { normal: [], anomalies: [] };
    return {
      normal: data.filter(p => !p.is_anomaly),
      anomalies: data.filter(p => p.is_anomaly)
    };
  }, [data]);

  return (
    <PanelState loading={loading} error={error}>
      <div className="w-full min-h-[300px] h-[300px]">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
            <XAxis
              dataKey="health_index"
              name="Health Index"
              type="number"
              stroke="var(--color-muted)"
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={{ stroke: 'var(--color-border)' }}
              label={{ value: 'Health Index', position: 'insideBottom', offset: -2, fill: 'var(--color-faint)', fontSize: 11 }}
            />
            <YAxis
              dataKey="velocity"
              name="Velocity"
              type="number"
              stroke="var(--color-muted)"
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
              label={{ value: 'Velocity', angle: -90, position: 'insideLeft', offset: 15, fill: 'var(--color-faint)', fontSize: 11 }}
            />
            <Tooltip content={<CustomTooltip />} />
            {/* Normal engines — colored by risk_state */}
            <Scatter
              name="Normal"
              data={normal}
              fill="var(--color-muted)"
              onClick={handlePointClick}
              className="cursor-pointer"
            >
              {normal.map((entry, i) => (
                <Cell
                  key={`normal-${i}`}
                  fill={STATE_COLOR[entry.risk_state] || 'var(--color-muted)'}
                  opacity={0.6}
                  className="cursor-pointer hover:opacity-100"
                />
              ))}
            </Scatter>
            {/* Anomalous engines — red diamonds */}
            <Scatter
              name="Anomaly"
              data={anomalies}
              fill="var(--color-critical)"
              shape="diamond"
              onClick={handlePointClick}
              className="cursor-pointer"
            >
              {anomalies.map((_, i) => (
                <Cell key={`anom-${i}`} fill="var(--color-critical)" opacity={1} className="cursor-pointer" />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-2 text-center text-xs text-muted">
        Each point is an engine — click one to open its drill-down.
      </div>
    </PanelState>
  );
}
