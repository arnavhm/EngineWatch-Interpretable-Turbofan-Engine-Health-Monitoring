import { useTopRisk } from '../hooks/useTopRisk';
import { STATE_TEXT, STATE_VAR } from '../stateColors';

interface TopRiskTableProps {
  datasetId: string;
  selectedEngine: number;
  onSelectEngine: (id: number) => void;
}

export default function TopRiskTable({ datasetId, selectedEngine, onSelectEngine }: TopRiskTableProps) {
  const { data, loading, error } = useTopRisk(datasetId);

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[150px]">
        <span className="text-muted text-sm border-l-2 border-critical pl-4 py-2">Error: {error}</span>
      </div>
    );
  }

  if (loading || !data) {
    return (
      <div className="flex flex-col gap-2 h-full animate-pulse mt-2">
        <div className="h-6 w-full bg-panel2 rounded-md mb-2"></div>
        <div className="h-10 w-full bg-panel2 rounded-md"></div>
        <div className="h-10 w-full bg-panel2 rounded-md"></div>
        <div className="h-10 w-full bg-panel2 rounded-md"></div>
        <div className="h-10 w-full bg-panel2 rounded-md"></div>
        <div className="h-10 w-full bg-panel2 rounded-md"></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden text-sm">
      <div className="grid grid-cols-[1fr_1fr_1fr_1.5fr] gap-2 px-3 py-2 text-[10px] text-muted font-mono tracking-widest uppercase border-b border-border">
        <div>Engine</div>
        <div>Risk</div>
        <div>RUL</div>
        <div>State</div>
      </div>
      
      <div className="flex flex-col flex-1 overflow-y-auto">
        {data.map((item) => {
          const isSelected = item.engine_id === selectedEngine;
          const colorClass = STATE_TEXT[item.risk_state];
          const colorVar = STATE_VAR[item.risk_state];
          
          return (
            <button
              key={item.engine_id}
              onClick={() => onSelectEngine(item.engine_id)}
              className={`grid grid-cols-[1fr_1fr_1fr_1.5fr] gap-2 px-3 py-3 items-center text-left font-mono transition-colors border-l-4 ${
                isSelected ? 'bg-panel2' : 'border-transparent hover:bg-panel2'
              }`}
              style={{
                borderLeftColor: isSelected ? colorVar : 'transparent'
              }}
            >
              <div className="font-bold text-text">#{item.engine_id}</div>
              <div className={`font-bold ${colorClass}`}>{item.risk_score.toFixed(3)}</div>
              <div className="text-text">{item.rul_cycles.toFixed(1)}</div>
              
              <div className="flex items-center gap-1.5">
                <div 
                  className="w-2 h-2 rounded-full" 
                  style={{ backgroundColor: colorVar }} 
                />
                <span className="text-xs font-bold tracking-wider" style={{ color: colorVar }}>
                  {item.risk_state.toUpperCase()}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
