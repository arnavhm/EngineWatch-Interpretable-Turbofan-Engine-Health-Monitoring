import { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceArea } from 'recharts';
import { useSensors } from '../hooks/useSensors';
import { useContributions } from '../hooks/useContributions';
import PanelState from './PanelState';

function rollingMean(values: number[], window: number): (number | null)[] {
  return values.map((_, i) => {
    if (i < window - 1) return null;
    const slice = values.slice(i - window + 1, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

interface SensorAccordionProps {
  engineId: number;
  datasetId: string;
}

export default function SensorAccordion({ engineId, datasetId }: SensorAccordionProps) {
  const { data: sData, loading: sLoading, error: sError } = useSensors(engineId, datasetId);
  const { data: cData, loading: cLoading, error: cError } = useContributions(engineId, datasetId);

  const [expandedKeys, setExpandedKeys] = useState<Set<string>>(new Set());

  const toggle = (key: string) => {
    const next = new Set(expandedKeys);
    if (next.has(key)) {
      next.delete(key);
    } else {
      next.add(key);
    }
    setExpandedKeys(next);
  };

  const sortedSensors = useMemo(() => {
    if (!sData || !sData.sensors) return [];
    
    const dominantText = cData?.dominant_driver_text || "";
    
    const entries = Object.entries(sData.sensors).map(([symbol, meta]) => ({
      symbol,
      ...meta
    }));

    entries.sort((a, b) => {
      const aIsDriver = dominantText.includes(a.symbol);
      const bIsDriver = dominantText.includes(b.symbol);
      
      if (aIsDriver && !bIsDriver) return -1;
      if (!aIsDriver && bIsDriver) return 1;
      
      if (a.confirmed && !b.confirmed) return -1;
      if (!a.confirmed && b.confirmed) return 1;
      
      return a.symbol.localeCompare(b.symbol);
    });

    return entries;
  }, [sData, cData]);

  if (sLoading || cLoading) {
    return <div className="animate-pulse bg-panel2 h-32 rounded border border-border"></div>;
  }

  if (sError || cError) {
    return <PanelState loading={false} error={sError || cError}><div/></PanelState>;
  }

  if (!sData) return null;

  const totalCycles = sData.cycles.length;
  const lateLifeStartCycle = sData.cycles[Math.floor(totalCycles * 0.75)];
  const maxCycle = sData.cycles[totalCycles - 1];

  return (
    <div className="flex flex-col gap-2">
      {sortedSensors.map(sensor => {
        const isExpanded = expandedKeys.has(sensor.symbol);
        
        return (
          <div key={sensor.symbol} className="border border-border rounded-lg bg-panel overflow-hidden">
            <button 
              className="w-full flex items-center justify-between p-3 hover:bg-panel2 transition-colors text-left"
              onClick={() => toggle(sensor.symbol)}
            >
              <div className="flex items-center gap-2">
                <span className="text-muted font-mono w-4">{isExpanded ? '▼' : '▶'}</span>
                <span className="font-bold font-mono text-text">{sensor.symbol}</span>
                <span className="text-muted hidden sm:inline">—</span>
                <span className="text-sm text-text hidden sm:inline">{sensor.descriptive_name}</span>
              </div>
              <div>
                {sensor.confirmed ? (
                  <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-[#3ECF8E]/20 text-[#3ECF8E] border border-[#3ECF8E]/30">
                    ✅ Confirmed · {sensor.module}
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2 py-1 rounded text-xs font-medium bg-panel2 text-muted border border-border">
                    Informational · {sensor.module}
                  </span>
                )}
              </div>
            </button>
            
            {isExpanded && (
              <div className="p-4 border-t border-border bg-panel2/30">
                <div className="h-48 w-full mb-4">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart margin={{ top: 5, right: 20, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" vertical={false} />
                      <XAxis 
                        dataKey="cycle" 
                        type="number"
                        domain={[sData.cycles[0], maxCycle]}
                        stroke="var(--color-muted)" 
                        tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                      />
                      <YAxis 
                        domain={['auto', 'auto']} 
                        stroke="var(--color-muted)"
                        tick={{ fill: 'var(--color-muted)', fontSize: 10, fontFamily: 'var(--font-mono)' }}
                      />
                      
                      <ReferenceArea x1={lateLifeStartCycle} x2={maxCycle} fill="rgba(255,255,255,0.03)" />
                      
                      {(() => {
                        const chartData = sData.cycles.map((cycle, i) => ({
                          cycle,
                          value: sensor.values[i],
                          rolling: rollingMean(sensor.values, 10)[i] ?? undefined
                        }));
                        
                        return (
                          <>
                            <Line
                              data={chartData}
                              type="monotone"
                              dataKey="value"
                              stroke="var(--color-muted)"
                              strokeWidth={1}
                              strokeOpacity={0.3}
                              dot={false}
                              isAnimationActive={false}
                            />
                            <Line
                              data={chartData}
                              type="monotone"
                              dataKey="rolling"
                              stroke="var(--color-accent)"
                              strokeWidth={2}
                              dot={false}
                              isAnimationActive={false}
                              connectNulls={false}
                            />
                          </>
                        );
                      })()}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                
                <div className="mb-2">
                  <p className="text-text font-medium">{sensor.layman_text}</p>
                </div>
                <div className="flex items-center gap-2 mb-2">
                  <span className={`font-bold ${sensor.signal_direction === 'falling' || sensor.signal_direction === 'decreasing' ? 'text-[#E0533A]' : 'text-[#3ECF8E]'}`}>
                    {sensor.signal_direction === 'falling' || sensor.signal_direction === 'decreasing' ? '↓' : '↑'}
                  </span>
                  <span className="text-sm text-muted">
                    expected to {sensor.signal_direction === 'falling' || sensor.signal_direction === 'decreasing' ? 'fall' : 'rise'} as the engine wears.
                  </span>
                </div>
                <div className="text-xs text-muted border-l-2 border-border pl-3 mt-3">
                  {sensor.explanation}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
