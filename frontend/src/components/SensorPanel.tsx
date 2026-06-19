import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { useSensors } from '../hooks/useSensors';
import PanelState from './PanelState';

interface SensorPanelProps {
  engineId: number;
  datasetId: string;
}

const SENSOR_COLORS: Record<string, string> = {
  T24: '#4DA3FF', T30: '#2DD4A7', T50: '#F5A524', P30: '#FF5A5F',
  Nf: '#A78BFA', Nc: '#F472B6', Ps30: '#34D399', phi: '#FBBF24',
  NRf: '#60A5FA', NRc: '#F87171', BPR: '#38BDF8', htBleed: '#C084FC',
  W31: '#FB923C', W32: '#818CF8'
};

export default function SensorPanel({ engineId, datasetId }: SensorPanelProps) {
  const { data, loading, error } = useSensors(engineId, datasetId);
  const sensorNames = useMemo(() => data ? Object.keys(data.sensors) : [], [data]);
  const [selected, setSelected] = useState<string[]>(['T50', 'Nc']);

  // Update selected when data loads if current selection is empty or invalid
  React.useEffect(() => {
    if (sensorNames.length > 0 && selected.every(s => !sensorNames.includes(s))) {
      setSelected([sensorNames[0]]);
    }
  }, [sensorNames]);

  const chartData = useMemo(() => {
    if (!data) return [];
    return data.cycles.map((cycle, i) => {
      const point: Record<string, number> = { cycle };
      for (const s of selected) {
        if (data.sensors[s]) point[s] = data.sensors[s][i];
      }
      return point;
    });
  }, [data, selected]);

  const toggleSensor = (name: string) => {
    setSelected(prev =>
      prev.includes(name) ? prev.filter(s => s !== name) : [...prev, name]
    );
  };

  return (
    <PanelState loading={loading} error={error}>
      <div className="flex flex-col gap-3">
        {/* Sensor selector */}
        <div className="flex flex-wrap gap-1.5">
          {sensorNames.map(name => (
            <button
              key={name}
              onClick={() => toggleSensor(name)}
              className={`px-2 py-0.5 text-xs font-mono rounded border transition-colors cursor-pointer ${
                selected.includes(name)
                  ? 'border-accent text-accent bg-accent/10'
                  : 'border-border text-muted hover:border-faint'
              }`}
            >
              {name}
            </button>
          ))}
        </div>

        {/* Chart */}
        <div className="w-full min-h-[250px] h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
              <XAxis
                dataKey="cycle"
                stroke="var(--color-muted)"
                tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
                tickLine={false}
                axisLine={{ stroke: 'var(--color-border)' }}
              />
              <YAxis
                stroke="var(--color-muted)"
                tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'var(--color-panel2)',
                  border: '1px solid var(--color-border)',
                  borderRadius: 6,
                  fontSize: 12,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-text)'
                }}
                labelStyle={{ color: 'var(--color-muted)' }}
              />
              {selected.map(name => (
                <Line
                  key={name}
                  type="monotone"
                  dataKey={name}
                  stroke={SENSOR_COLORS[name] || 'var(--color-accent)'}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </PanelState>
  );
}
