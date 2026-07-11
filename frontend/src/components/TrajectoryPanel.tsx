import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceArea, ReferenceLine, ReferenceDot, ResponsiveContainer } from 'recharts';
import PanelState from './PanelState';
import type { TrajectoryResponse } from '../types';

function rollingMean(values: number[], window: number): (number | null)[] {
  return values.map((_, i) => {
    if (i < window - 1) return null;
    const slice = values.slice(i - window + 1, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

interface TrajectoryPanelProps {
  data: TrajectoryResponse | null;
  loading: boolean;
  error: string | null;
}

export default function TrajectoryPanel({ data, loading, error }: TrajectoryPanelProps) {

  // Transform data into Recharts format, with 10-cycle rolling mean
  const chartData = React.useMemo(() => {
    if (!data) return [];
    const rollingValues = rollingMean(data.health_index, 10);
    return data.cycles.map((cycle, i) => ({
      cycle,
      health_index: data.health_index[i],
      hi_rolling_mean: rollingValues[i] ?? undefined,
    }));
  }, [data]);

  const caption = React.useMemo(() => {
    if (!data || data.cycles.length === 0) return null;
    const hi = data.health_index;
    const cycles = data.cycles;
    const velocity = data.velocity;
    const lastIndex = hi.length - 1;
    
    const firstCritIndex = hi.findIndex(v => v < 0.30);
    const critStr = firstCritIndex !== -1 ? `, crossing the critical threshold at cycle ${cycles[firstCritIndex]}` : '';
    
    const lastVel = velocity[lastIndex];
    const trendStr = lastVel < 0 ? 'still falling' : 'now stabilizing';
    
    return `Health index declined from ${hi[0].toFixed(2)} to ${hi[lastIndex].toFixed(2)} over ${cycles.length} cycles${critStr}; ${trendStr} (latest velocity ${lastVel.toExponential(1)}/cycle).`;
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

  const lastPoint = chartData[chartData.length - 1];

  return (
    <PanelState loading={loading} error={error}>
      <div className="w-full h-full flex flex-col min-h-[250px] relative">
        <div className="flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
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

              {/* Critical threshold line */}
              <ReferenceLine
                y={0.3}
                stroke="var(--color-critical)"
                strokeDasharray="5 3"
                strokeWidth={1.5}
                label={{ value: "Critical", position: "insideTopRight", fill: "var(--color-critical)", fontSize: 10 }}
              />

              {lastPoint && (
                <ReferenceLine
                  x={lastPoint.cycle}
                  stroke="var(--color-muted)"
                  strokeDasharray="3 3"
                  strokeOpacity={0.5}
                />
              )}

              {lastPoint && (
                <ReferenceDot 
                  x={lastPoint.cycle} 
                  y={lastPoint.health_index} 
                  r={5} 
                  fill="var(--color-accent)" 
                  stroke="var(--color-panel)" 
                  strokeWidth={2}
                  label={{ value: lastPoint.health_index.toFixed(3), position: "top", fill: "var(--color-text)", fontSize: 11, fontFamily: "var(--font-mono)" }}
                />
              )}

              <Line
                type="monotone"
                dataKey="health_index"
                stroke="var(--color-accent)"
                strokeWidth={2}
                strokeOpacity={0.35}
                dot={false}
                activeDot={{ r: 4, fill: 'var(--color-accent)', stroke: 'var(--color-panel)' }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="hi_rolling_mean"
                name="Rolling Mean (10 cy)"
                stroke="var(--color-accent)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        {caption && (
          <div className="mt-4 text-center text-sm text-muted">
            {caption}
          </div>
        )}
      </div>
    </PanelState>
  );
}
