import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import PanelState from './PanelState';
import type { TrajectoryResponse } from '../types';

interface VariabilityPanelProps {
  data: TrajectoryResponse | null;
  loading: boolean;
  error: string | null;
}

export default function VariabilityPanel({ data, loading, error }: VariabilityPanelProps) {

  const chartData = React.useMemo(() => {
    if (!data) return [];
    return data.cycles.map((cycle, i) => ({
      cycle,
      variability: data.variability[i]
    }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-panel2 border border-border p-2 rounded shadow-md text-xs font-mono">
          <p className="text-muted mb-1">Cycle: {label}</p>
          <p className="text-text">
            Variability: <span className="text-degrading font-bold">{payload[0].value.toFixed(4)}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <PanelState loading={loading} error={error}>
      <div className="w-full h-full flex flex-col min-h-[250px] relative">
        <div className="flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorVariability" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--color-degrading)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--color-degrading)" stopOpacity={0.0}/>
                </linearGradient>
              </defs>
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
                tickFormatter={(val) => val.toFixed(3)}
              />
              <Tooltip content={<CustomTooltip />} />
              
              <Area
                type="monotone"
                dataKey="variability"
                stroke="var(--color-degrading)"
                fillOpacity={1}
                fill="url(#colorVariability)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-center text-sm text-muted">
          How erratic the health signal is. Rising = less stable, harder to predict.
        </div>
      </div>
    </PanelState>
  );
}
