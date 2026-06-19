import { usePredict } from '../hooks/usePredict';
import { STATE_TEXT } from '../stateColors';
import PanelState from './PanelState';

interface RiskDialProps {
  engineId: number;
  datasetId: string;
}

export default function RiskDial({ engineId, datasetId }: RiskDialProps) {
  const { data, loading, error } = usePredict(engineId, datasetId);

  return (
    <PanelState loading={loading || !data} error={error}>
      {data && (() => {
        const { risk_score, risk_state } = data;
        const colorClass = STATE_TEXT[risk_state];
        
        const radius = 80;
        const circumference = 2 * Math.PI * radius; 
        const arcLength = 0.75 * circumference; 
        
        const healthyLen = 0.46 * arcLength; 
        const degradingLen = 0.20 * arcLength; 
        const criticalLen = 0.34 * arcLength; 
        
        const baseRot = 135;
        const healthyRot = baseRot;
        const degradingRot = baseRot + (0.46 * 270);
        const criticalRot = baseRot + (0.66 * 270);

        const needleRot = baseRot + (risk_score * 270);

        return (
          <div className="flex flex-col items-center justify-center w-full h-full p-4">
            <div className="relative w-[200px] h-[200px]">
              <svg viewBox="0 0 200 200" className="w-full h-full overflow-visible">
                {/* Track background */}
                <circle 
                  cx="100" cy="100" r={radius} 
                  fill="none" 
                  stroke="var(--color-border)" 
                  strokeWidth="12" 
                  strokeDasharray={`${arcLength} ${circumference}`} 
                  transform={`rotate(${baseRot} 100 100)`} 
                  strokeLinecap="round" 
                />
                
                {/* Healthy Zone */}
                <circle 
                  cx="100" cy="100" r={radius} 
                  fill="none" 
                  stroke="var(--color-healthy)" 
                  strokeWidth="12" 
                  strokeDasharray={`${healthyLen} ${circumference}`} 
                  transform={`rotate(${healthyRot} 100 100)`} 
                />

                {/* Degrading Zone */}
                <circle 
                  cx="100" cy="100" r={radius} 
                  fill="none" 
                  stroke="var(--color-degrading)" 
                  strokeWidth="12" 
                  strokeDasharray={`${degradingLen} ${circumference}`} 
                  transform={`rotate(${degradingRot} 100 100)`} 
                />

                {/* Critical Zone */}
                <circle 
                  cx="100" cy="100" r={radius} 
                  fill="none" 
                  stroke="var(--color-critical)" 
                  strokeWidth="12" 
                  strokeDasharray={`${criticalLen} ${circumference}`} 
                  transform={`rotate(${criticalRot} 100 100)`} 
                />
                
                <circle 
                  cx="100" cy="100" r={radius} 
                  fill="none" 
                  stroke="var(--color-critical)" 
                  strokeWidth="12" 
                  strokeDasharray={`0.1 ${circumference}`} 
                  transform={`rotate(${baseRot + 270} 100 100)`} 
                  strokeLinecap="round" 
                />
                <circle 
                  cx="100" cy="100" r={radius} 
                  fill="none" 
                  stroke="var(--color-healthy)" 
                  strokeWidth="12" 
                  strokeDasharray={`0.1 ${circumference}`} 
                  transform={`rotate(${baseRot} 100 100)`} 
                  strokeLinecap="round" 
                />

                {/* Needle */}
                <g transform={`rotate(${needleRot} 100 100)`}>
                  <line 
                    x1="100" y1="100" x2="165" y2="100" 
                    stroke="var(--color-text)" 
                    strokeWidth="4" 
                    strokeLinecap="round" 
                  />
                  <circle cx="100" cy="100" r="6" fill="var(--color-text)" />
                </g>
              </svg>

              <div className="absolute inset-0 flex flex-col items-center justify-center mt-12">
                <span className={`text-4xl font-mono font-bold ${colorClass} leading-none`}>
                  {risk_score.toFixed(2)}
                </span>
                <span className="text-[10px] text-muted font-mono tracking-widest mt-1">RISK</span>
              </div>
            </div>

            <div className="flex gap-4 mt-6">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-healthy"></div>
                <span className="text-[10px] text-muted font-mono">0.0–0.46</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-degrading"></div>
                <span className="text-[10px] text-muted font-mono">0.46–0.66</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-critical"></div>
                <span className="text-[10px] text-muted font-mono">&gt;0.66</span>
              </div>
            </div>
          </div>
        );
      })()}
    </PanelState>
  );
}
