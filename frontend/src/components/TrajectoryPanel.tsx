import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceArea, ResponsiveContainer } from 'recharts';
import { useTrajectory } from '../hooks/useTrajectory';
import PanelState from './PanelState';

interface TrajectoryPanelProps {
  engineId: number;
  datasetId: string;
}

export default function TrajectoryPanel({ engineId, datasetId }: TrajectoryPanelProps) {
  const { data, loading, error } = useTrajectory(engineId, datasetId);

  // Transform data into Recharts format
  const chartData = React.useMemo(() => {
    if (!data) return [];
    return data.cycles.map((cycle, i) => ({
      cycle,
      health_index: data.health_index[i]
    }));
  }, [data]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-panel2 border border-border p-2 rounded shadow-md text-xs font-mono">
          <p className="text-muted mb-1">Cycle: {label}</p>
          <p className="text-text">
            HI: <span className="text-accent font-bold">{payload[0].value.toFixed(3)}</span>
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <PanelState loading={loading} error={error}>
      <div className="w-full h-full min-h-[250px] relative">
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
              domain={[0, 1]} 
              stroke="var(--color-muted)"
              tick={{ fill: 'var(--color-muted)', fontSize: 12, fontFamily: 'var(--font-mono)' }}
              tickLine={false}
              axisLine={false}
              tickCount={5}
            />
            <Tooltip content={<CustomTooltip />} />
            
            {/* Reference Zones */}
            <ReferenceArea y1={0.66} y2={1.0} fill="var(--color-healthy)" fillOpacity={0.05} />
            <ReferenceArea y1={0.46} y2={0.66} fill="var(--color-degrading)" fillOpacity={0.05} />
            <ReferenceArea y1={0} y2={0.46} fill="var(--color-critical)" fillOpacity={0.05} />
            
            <Line 
              type="monotone" 
              dataKey="health_index" 
              stroke="var(--color-accent)" 
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, fill: 'var(--color-accent)', stroke: 'var(--color-panel)' }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </PanelState>
  );
}
