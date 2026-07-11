import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import PanelState from './PanelState';
import type { TrajectoryResponse } from '../types';

interface VelocityPanelProps {
  data: TrajectoryResponse | null;
  loading: boolean;
  error: string | null;
}

export default function VelocityPanel({ data, loading, error }: VelocityPanelProps) {

  const chartData = React.useMemo(() => {
    if (!data) return [];
    return data.cycles.map((cycle, i) => ({
      cycle,
      velocity: data.velocity[i]
    }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-panel2 border border-border p-2 rounded shadow-md text-xs font-mono">
          <p className="text-muted mb-1">Cycle: {label}</p>
          <p className="text-text">
            Velocity: <span className="text-accent font-bold">{payload[0].value.toExponential(2)}</span>
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
                <linearGradient id="colorVelocity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--color-accent)" stopOpacity={0.4}/>
                  <stop offset="95%" stopColor="var(--color-accent)" stopOpacity={0.0}/>
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
                tickFormatter={(val) => val.toExponential(0)}
              />
              <Tooltip content={<CustomTooltip />} />
              
              <ReferenceLine y={0} stroke="var(--color-muted)" strokeDasharray="3 3" />
              
              <Area
                type="monotone"
                dataKey="velocity"
                stroke="var(--color-accent)"
                fillOpacity={1}
                fill="url(#colorVelocity)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-4 text-center text-sm text-muted">
          How fast health is dropping. Steeper negative = faster degradation.
        </div>
      </div>
    </PanelState>
  );
}
