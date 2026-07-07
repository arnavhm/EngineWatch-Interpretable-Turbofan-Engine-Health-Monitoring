import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import PanelState from './PanelState';
import type { FleetAnalyticsResponse } from '../types';

interface RiskHistogramChartProps {
  data: FleetAnalyticsResponse | null;
  loading: boolean;
  error: string | null;
}

export default function RiskHistogramChart({ data, loading, error }: RiskHistogramChartProps) {

  return (
    <PanelState loading={loading || !data} error={error}>
      {data && (() => {
        const chartData = data.risk_histogram.map(bin => ({
          name: `${bin.bin_start.toFixed(1)}-${bin.bin_end.toFixed(1)}`,
          count: bin.count
        }));

        return (
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
                <XAxis 
                  dataKey="name" 
                  stroke="var(--color-muted)" 
                  fontSize={12} 
                  tickLine={false} 
                  axisLine={false}
                  angle={-45}
                  textAnchor="end"
                  height={60}
                />
                <YAxis 
                  stroke="var(--color-muted)" 
                  fontSize={12} 
                  tickLine={false} 
                  axisLine={false} 
                  allowDecimals={false}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--color-panel2)', borderColor: 'var(--color-border)', borderRadius: '8px' }}
                  itemStyle={{ color: 'var(--color-text)' }}
                  labelStyle={{ color: 'var(--color-muted)', marginBottom: '4px' }}
                  cursor={{ fill: 'var(--color-border)', opacity: 0.4 }}
                />
                <Bar dataKey="count" fill="var(--color-accent)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      })()}
    </PanelState>
  );
}
