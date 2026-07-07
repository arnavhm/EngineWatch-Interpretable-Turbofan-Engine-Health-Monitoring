import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
import PanelState from './PanelState';
import type { FleetAnalyticsResponse } from '../types';

interface FleetTrendChartProps {
  data: FleetAnalyticsResponse | null;
  loading: boolean;
  error: string | null;
}

export default function FleetTrendChart({ data, loading, error }: FleetTrendChartProps) {

  return (
    <PanelState loading={loading || !data} error={error}>
      {data && (() => {
        const chartData = data.risk_trend.map(decile => ({
          name: `${decile.life_pct_bin * 10}-${(decile.life_pct_bin + 1) * 10}%`,
          mean_risk_score: decile.mean_risk_score,
          n_engines_contributing: decile.n_engines_contributing
        }));

        return (
          <div className="w-full h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
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
                  domain={[0, 1]}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'var(--color-panel2)', borderColor: 'var(--color-border)', borderRadius: '8px' }}
                  itemStyle={{ color: 'var(--color-text)' }}
                  labelStyle={{ color: 'var(--color-muted)', marginBottom: '4px' }}
                  formatter={(value: any, name: any, props: any) => {
                    if (name === 'mean_risk_score' && typeof value === 'number') {
                      return [
                        `${value.toFixed(3)} (based on ${props.payload.n_engines_contributing} engines observed this late)`, 
                        'Mean Risk'
                      ];
                    }
                    return [value, name];
                  }}
                />
                <Line 
                  type="monotone" 
                  dataKey="mean_risk_score" 
                  stroke="var(--color-critical)" 
                  strokeWidth={2}
                  dot={{ r: 3, fill: 'var(--color-critical)' }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        );
      })()}
    </PanelState>
  );
}
