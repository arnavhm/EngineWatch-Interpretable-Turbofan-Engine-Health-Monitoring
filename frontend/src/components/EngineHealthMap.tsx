import { useState } from 'react';
import { useContributions } from '../hooks/useContributions';
import PanelState from './PanelState';

interface EngineHealthMapProps {
  engineId: number;
  datasetId: string;
}

const STATIC_BLURBS: Record<string, any> = {
  fan: { name: 'Fan', spool: 'N1 · LP spool', blurb: 'About eleven wide-chord blades on the low-pressure spool. The air it drives through the bypass duct supplies most of the engine’s thrust.' },
  lpc: { name: 'LPC', spool: 'N1 · LP spool', blurb: 'Five axial stages (the booster) that pre-compress core flow ahead of the HPC. Shares the LP spool with the fan and LPT.' },
  hpc: { name: 'HPC', spool: 'N2 · HP spool', blurb: 'Eight dense stages delivering the bulk of the pressure rise. Interstage bleed ports tap air for turbine cooling and cabin systems.' },
  comb: { name: 'Combustor', spool: 'Static structure', blurb: 'Fuel is injected and burned here — the hottest gas in the engine and the cold-to-hot transition point of the cycle.' },
  hpt: { name: 'HPT', spool: 'N2 · HP spool', blurb: 'Three stages that extract work to drive the HPC. Blades are film-cooled with bleed air ducted forward from the compressor.' },
  lpt: { name: 'LPT', spool: 'N1 · LP spool', blurb: 'Five progressively larger stages driving the fan and LPC through the LP spool, which runs the full length of the core.' },
  nozzle: { name: 'Core nozzle', spool: 'Static structure', blurb: 'Converging nozzle and tail cone that accelerate the spent core gas, adding residual core thrust to the bypass stream.' },
  bypass: { name: 'Bypass duct', spool: 'Cold flow path', blurb: 'The annular channel between nacelle and core carries most of the mass flow around the engine — the dominant thrust source.' }
};

const SVG_PATHS: Record<string, string> = {
  fan: "M45,88 L108,72 L108,248 L45,232 Z",
  lpc: "M108,100 L180,100 L180,220 L108,220 Z",
  hpc: "M180,100 L260,100 L260,220 L180,220 Z",
  comb: "M260,100 L315,103 L315,217 L260,220 Z",
  hpt: "M315,103 L360,105 L360,215 L315,217 Z",
  lpt: "M360,105 L450,110 L450,210 L360,215 Z",
  nozzle: "M450,110 L558,132 L566,142 L566,178 L558,188 L450,210 Z",
  bypass: "M108,78 L440,86 L510,104 L546,124 L558,133 L450,110 L360,105 L315,103 L260,100 L180,100 L108,100 Z M108,242 L440,234 L510,216 L546,196 L558,187 L450,210 L360,215 L315,217 L260,220 L180,220 L108,220 Z"
};

const API_MAP: Record<string, string> = {
  fan: 'fan',
  lpc: 'lpc',
  hpc: 'hpc',
  comb: 'hpc',
  hpt: 'hpt',
  lpt: 'lpt',
  nozzle: '',
  bypass: 'bypass'
};

export default function EngineHealthMap({ engineId, datasetId }: EngineHealthMapProps) {
  const { data, loading, error } = useContributions(engineId, datasetId);
  const [hoveredZone, setHoveredZone] = useState<string | null>(null);

  const getZoneColor = (zone: string, isHovered: boolean) => {
    if (!data) return { fill: 'transparent', stroke: 'transparent' };
    const apiKey = API_MAP[zone];
    
    // Base colors
    let r = 111, g = 168, b = 216; // default blueish
    let alpha = 0;
    let stroke = 'transparent';
    
    if (apiKey) {
      const mod = data.modules.find(m => m.module.toLowerCase() === apiKey.toLowerCase());
      if (mod) {
        // Color by direction
        if (mod.direction === 'critical') {
          r = 224; g = 83; b = 58; // #E0533A
        } else if (mod.direction === 'healthy') {
          r = 62; g = 207; b = 142; // #3ECF8E
        } else {
          r = 224; g = 169; b = 58; // #E0A93A degrading
        }
        
        alpha = mod.norm_magnitude * 0.6; // Scale opacity by severity
        
        if (data.dominant_module.toLowerCase() === apiKey.toLowerCase()) {
          stroke = `rgba(${r},${g},${b}, 1)`;
          alpha = Math.max(alpha, 0.3); // Ensure visible
        }
      }
    }
    
    if (isHovered) {
      alpha = Math.min(alpha + 0.2, 0.8);
      stroke = stroke === 'transparent' ? 'rgba(255,255,255,0.3)' : stroke;
    }
    
    return {
      fill: `rgba(${r},${g},${b},${alpha})`,
      stroke,
      strokeWidth: 1.4,
      cursor: 'pointer',
      transition: 'fill .15s ease, stroke .15s ease'
    };
  };

  const renderPaths = () => {
    return Object.entries(SVG_PATHS).map(([zone, pathData]) => (
      <path
        key={zone}
        d={pathData}
        style={getZoneColor(zone, hoveredZone === zone)}
        onMouseEnter={() => setHoveredZone(zone)}
        onMouseLeave={() => setHoveredZone(null)}
      />
    ));
  };

  // Find info for the details panel
  let panelContent = null;
  if (hoveredZone) {
    const staticInfo = STATIC_BLURBS[hoveredZone];
    const apiKey = API_MAP[hoveredZone];
    const mod = apiKey && data ? data.modules.find(m => m.module.toLowerCase() === apiKey.toLowerCase()) : null;
    
    let isDominant = false;
    if (mod && data) {
       isDominant = data.dominant_module.toLowerCase() === mod.module.toLowerCase();
    }
    
    panelContent = (
      <div className="flex flex-col gap-2 p-4 h-full bg-panel2/30">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-accent">{mod ? mod.display_name : staticInfo.name}</span>
          <span className="text-xs text-muted font-mono bg-panel border border-border px-2 py-1 rounded">{staticInfo.spool}</span>
          {mod && (
            <span className={`text-xs font-bold px-2 py-1 rounded ${mod.direction === 'critical' ? 'bg-[#E0533A]/20 text-[#E0533A]' : (mod.direction === 'healthy' ? 'bg-[#3ECF8E]/20 text-[#3ECF8E]' : 'bg-[#E0A93A]/20 text-[#E0A93A]')}`}>
              {mod.direction.toUpperCase()}
            </span>
          )}
        </div>
        
        {isDominant && data && (
          <div className="text-xs text-[#E0533A] font-medium border-l-2 border-[#E0533A] pl-2 mb-2">
            {data.dominant_driver_text}
          </div>
        )}
        
        {mod && mod.active_sensors && mod.active_sensors.length > 0 ? (
          <div className="mt-2">
            <div className="text-xs text-muted mb-1 font-bold uppercase tracking-wider">Active Sensors</div>
            <div className="grid grid-cols-1 gap-1">
              {mod.active_sensors.map(s => (
                <div key={s.symbol} className="flex items-center justify-between text-sm bg-panel border border-border p-1.5 rounded">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold">{s.symbol}</span>
                    <span className="text-muted text-xs truncate max-w-[150px]">{s.description}</span>
                  </div>
                  <span className={`font-mono font-bold ${s.signed_contribution < 0 ? 'text-[#E0533A]' : 'text-[#3ECF8E]'}`}>
                    {s.signed_contribution > 0 ? '+' : ''}{s.signed_contribution.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <div className="text-sm text-muted mt-2 border-l-2 border-border pl-3">
            {staticInfo.blurb}
          </div>
        )}
      </div>
    );
  } else {
    panelContent = (
      <div className="flex items-center justify-center h-full text-muted text-sm font-mono opacity-50 p-6 text-center">
        Hover a section of the engine to inspect its live health contribution.
      </div>
    );
  }

  return (
    <PanelState loading={loading} error={error}>
      <div className="w-full flex flex-col gap-4">
        <div className="w-full rounded-lg bg-[#080E18] overflow-hidden border border-[#16202E]">
                <svg width="100%" viewBox="0 0 690 330" role="img" xmlns="http://www.w3.org/2000/svg" style={{ display: 'block' }}>
        <title>2-spool turbofan engine cross-section</title>
        <desc>Side-profile technical cutaway of a generic 2-spool turbofan engine showing fan, LPC, HPC, combustor, HPT, LPT and bypass duct. Cold section in steel blue, hot section in amber and red.</desc>
        <defs>
          <linearGradient id="hot" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#E07020"></stop>
            <stop offset="55%" stopColor="#C03808"></stop>
            <stop offset="100%" stopColor="#8B2000"></stop>
          </linearGradient>
          <filter id="heat" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="7"></feGaussianBlur>
          </filter>
        </defs>

        <rect width="690" height="330" fill="#080E18"></rect>

        <polygon points="45,88 108,72 440,80 510,98 552,121 566,141 569,160 566,179 552,199 510,222 440,240 108,248 45,232" fill="#263040" stroke="#3D4E60" strokeWidth="1"></polygon>
        <ellipse cx="46" cy="160" rx="10" ry="72" fill="#1E2A3A" stroke="#3D4E60" strokeWidth="1.2"></ellipse>
        <ellipse cx="46" cy="160" rx="4" ry="28" fill="#1A2E48"></ellipse>
        <polygon points="46,87 108,78 108,242 46,233" fill="#12213A"></polygon>
        <polygon points="108,78 440,86 510,104 546,124 558,133 450,110 360,105 315,103 260,100 180,100 108,100" fill="#12213A"></polygon>
        <polygon points="108,242 440,234 510,216 546,196 558,187 450,210 360,215 315,217 260,220 180,220 108,220" fill="#12213A"></polygon>

        <polygon points="108,100 180,100 180,220 108,220" fill="#1E3A5C"></polygon>
        <polygon points="108,100 180,100 180,106 108,106" fill="#264A72"></polygon>
        <polygon points="108,214 180,214 180,220 108,220" fill="#264A72"></polygon>
        <polygon points="180,100 260,100 260,220 180,220" fill="#152E4A"></polygon>
        <polygon points="180,100 260,100 260,106 180,106" fill="#1E3A5C"></polygon>
        <polygon points="180,214 260,214 260,220 180,220" fill="#1E3A5C"></polygon>

        <ellipse cx="287" cy="160" rx="34" ry="58" fill="#E07020" opacity="0.30" filter="url(#heat)" ></ellipse>

        <polygon points="260,100 315,103 315,217 260,220" fill="#2A1808"></polygon>
        <polygon points="263,112 312,114 312,206 263,208" fill="url(#hot)"></polygon>
        <polygon points="267,124 308,126 308,194 267,196" fill="#F08030"></polygon>
        <circle cx="280" cy="118" r="3" fill="#FFB060"></circle>
        <circle cx="295" cy="119" r="3" fill="#FFB060"></circle>
        <circle cx="280" cy="202" r="3" fill="#FFB060"></circle>
        <circle cx="295" cy="201" r="3" fill="#FFB060"></circle>

        <polygon points="315,103 360,105 360,215 315,217" fill="#5C2810"></polygon>
        <polygon points="315,103 360,105 360,111 315,109" fill="#7B3A18"></polygon>
        <polygon points="315,211 360,209 360,215 315,217" fill="#7B3A18"></polygon>
        <polygon points="360,105 450,110 450,210 360,215" fill="#402810"></polygon>
        <polygon points="360,105 450,110 450,116 360,111" fill="#5A3818"></polygon>
        <polygon points="360,209 450,204 450,210 360,215" fill="#5A3818"></polygon>
        <polygon points="450,110 558,132 566,142 566,178 558,188 450,210" fill="#1E2C3C"></polygon>
        <polygon points="450,122 554,138 560,150 560,170 554,182 450,198" fill="#180A00"></polygon>

        <rect x="108" y="156" width="450" height="8" fill="#2C3A48" rx="1"></rect>
        <rect x="108" y="158" width="450" height="4" fill="#4B5563" rx="1"></rect>
        <rect x="180" y="151" width="180" height="18" fill="#1A2533" opacity="0.65" rx="2"></rect>
        <rect x="180" y="150" width="180" height="20" fill="none" stroke="#5A6A7A" strokeWidth="1" rx="2"></rect>
        <text x="116" y="182" fill="#6E7E90" fontSize="7" fontFamily="'IBM Plex Mono', monospace">N1 — LP shaft</text>
        <text x="214" y="147" fill="#8FA0B2" fontSize="7" fontFamily="'IBM Plex Mono', monospace">N2 — HP shaft</text>

        <polyline points="46,88 108,72 440,80 510,98 552,121 566,141 569,160" fill="none" stroke="#4A5C70" strokeWidth="0.8"></polyline>
        <polyline points="46,232 108,248 440,240 510,222 552,199 566,179 569,160" fill="none" stroke="#4A5C70" strokeWidth="0.8"></polyline>
        <polyline points="108,78 440,86 510,104 546,124 558,133" fill="none" stroke="#2E4050" strokeWidth="0.5"></polyline>
        <polyline points="108,242 440,234 510,216 546,196 558,187" fill="none" stroke="#2E4050" strokeWidth="0.5"></polyline>
        <polyline points="108,100 180,100 260,100 315,103 360,105 450,110 558,132" fill="none" stroke="#5A6A7A" strokeWidth="0.8"></polyline>
        <polyline points="108,220 180,220 260,220 315,217 360,215 450,210 558,188" fill="none" stroke="#5A6A7A" strokeWidth="0.8"></polyline>
        <polyline points="108,116 180,116 260,116 263,112 312,114 315,118 360,120 450,124 558,140" fill="none" stroke="#8A9AAA" strokeWidth="0.5" strokeDasharray="3,2"></polyline>
        <polyline points="108,204 180,204 260,204 263,208 312,206 315,202 360,200 450,196 558,180" fill="none" stroke="#8A9AAA" strokeWidth="0.5" strokeDasharray="3,2"></polyline>

        <ellipse cx="108" cy="160" rx="5" ry="62" fill="#2E3D4E" stroke="#4A5C70" strokeWidth="0.8"></ellipse>
        <g stroke="#8AAABB" strokeWidth="2.5" strokeLinecap="round">
          <line x1="108" y1="98" x2="70" y2="80"></line>
          <line x1="108" y1="111" x2="62" y2="97"></line>
          <line x1="108" y1="124" x2="58" y2="113"></line>
          <line x1="108" y1="137" x2="56" y2="130"></line>
          <line x1="108" y1="150" x2="56" y2="148"></line>
          <line x1="108" y1="160" x2="55" y2="160"></line>
          <line x1="108" y1="170" x2="56" y2="172"></line>
          <line x1="108" y1="183" x2="56" y2="190"></line>
          <line x1="108" y1="196" x2="58" y2="207"></line>
          <line x1="108" y1="209" x2="62" y2="223"></line>
          <line x1="108" y1="222" x2="70" y2="240"></line>
        </g>
        <ellipse cx="108" cy="160" rx="3.5" ry="22" fill="#5A6A7A"></ellipse>

        <g stroke="#4A7CAC" strokeWidth="1.8" strokeLinecap="round">
          <line x1="124" y1="100" x2="121" y2="115"></line> <line x1="124" y1="220" x2="121" y2="205"></line>
          <line x1="137" y1="100" x2="134" y2="115"></line> <line x1="137" y1="220" x2="134" y2="205"></line>
          <line x1="150" y1="100" x2="147" y2="115"></line> <line x1="150" y1="220" x2="147" y2="205"></line>
          <line x1="163" y1="100" x2="160" y2="114"></line> <line x1="163" y1="220" x2="160" y2="206"></line>
          <line x1="173" y1="100" x2="170" y2="114"></line> <line x1="173" y1="220" x2="170" y2="206"></line>
        </g>
        <g stroke="#3A6080" strokeWidth="1.2" strokeLinecap="round">
          <line x1="190" y1="100" x2="188" y2="113"></line> <line x1="190" y1="220" x2="188" y2="207"></line>
          <line x1="199" y1="100" x2="197" y2="113"></line> <line x1="199" y1="220" x2="197" y2="207"></line>
          <line x1="208" y1="100" x2="206" y2="112"></line> <line x1="208" y1="220" x2="206" y2="208"></line>
          <line x1="217" y1="100" x2="215" y2="112"></line> <line x1="217" y1="220" x2="215" y2="208"></line>
          <line x1="226" y1="100" x2="224" y2="112"></line> <line x1="226" y1="220" x2="224" y2="208"></line>
          <line x1="235" y1="100" x2="233" y2="111"></line> <line x1="235" y1="220" x2="233" y2="209"></line>
          <line x1="244" y1="100" x2="242" y2="111"></line> <line x1="244" y1="220" x2="242" y2="209"></line>
          <line x1="253" y1="100" x2="251" y2="111"></line> <line x1="253" y1="220" x2="251" y2="209"></line>
        </g>
        <g stroke="#C86028" strokeWidth="2.2" strokeLinecap="round">
          <line x1="326" y1="103" x2="330" y2="118"></line> <line x1="326" y1="217" x2="330" y2="202"></line>
          <line x1="338" y1="104" x2="342" y2="119"></line> <line x1="338" y1="216" x2="342" y2="201"></line>
          <line x1="350" y1="104" x2="354" y2="119"></line> <line x1="350" y1="216" x2="354" y2="201"></line>
        </g>
        <g stroke="#885018" strokeWidth="2" strokeLinecap="round">
          <line x1="372" y1="105" x2="377" y2="121"></line> <line x1="372" y1="215" x2="377" y2="199"></line>
          <line x1="388" y1="106" x2="393" y2="122"></line> <line x1="388" y1="214" x2="393" y2="198"></line>
          <line x1="404" y1="107" x2="409" y2="123"></line> <line x1="404" y1="213" x2="409" y2="197"></line>
          <line x1="420" y1="108" x2="425" y2="124"></line> <line x1="420" y1="212" x2="425" y2="196"></line>
          <line x1="436" y1="109" x2="441" y2="124"></line> <line x1="436" y1="211" x2="441" y2="196"></line>
        </g>

        <g stroke="#3A4A5A" strokeWidth="0.6" strokeDasharray="4,2">
          <line x1="180" y1="100" x2="180" y2="220"></line>
          <line x1="260" y1="100" x2="260" y2="220"></line>
          <line x1="315" y1="103" x2="315" y2="217"></line>
          <line x1="360" y1="105" x2="360" y2="215"></line>
          <line x1="450" y1="110" x2="450" y2="210"></line>
        </g>

        <ellipse cx="568" cy="160" rx="7" ry="18" fill="#1E2C3C" stroke="#4A5C70" strokeWidth="0.8"></ellipse>
        <ellipse cx="568" cy="160" rx="3" ry="9" fill="#120800"></ellipse>

        <g  fontFamily="'IBM Plex Mono', monospace">
          <polygon points="104,89 116,96 104,103" fill="#3D4E60" stroke="#5A6A7A" strokeWidth="0.6"></polygon>
          <line x1="110" y1="84" x2="118" y2="76" stroke="#5A6A7A" strokeWidth="0.5"></line>
          <text x="120" y="74" fill="#7E8EA0" fontSize="7">splitter</text>
          <g stroke="#6FA8D8" strokeWidth="1" strokeLinecap="round">
            <line x1="214" y1="100" x2="212" y2="92"></line>
            <line x1="228" y1="100" x2="226" y2="92"></line>
            <line x1="242" y1="100" x2="240" y2="92"></line>
          </g>
          <text x="228" y="88" fill="#6FA8D8" fontSize="7" textAnchor="middle">bleed air</text>
          <g stroke="#6FA8D8" strokeWidth="0.6" strokeDasharray="2,2">
            <line x1="320" y1="152" x2="332" y2="126"></line>
            <line x1="320" y1="168" x2="332" y2="194"></line>
          </g>
          <text x="336" y="150" fill="#6FA8D8" fontSize="7">cooling air</text>
        </g>

        <g fontFamily="'IBM Plex Sans', sans-serif" fontSize="12" fontWeight="500" fill="#C3D2E2">
          <line x1="70" y1="68" x2="70" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="70" y="44" textAnchor="middle">Fan</text>
          <line x1="144" y1="68" x2="144" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="144" y="44" textAnchor="middle">LPC</text>
          <line x1="220" y1="68" x2="220" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="220" y="44" textAnchor="middle">HPC</text>
          <line x1="287" y1="68" x2="287" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="287" y="44" textAnchor="middle">Combustor</text>
          <line x1="337" y1="68" x2="337" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="337" y="44" textAnchor="middle">HPT</text>
          <line x1="405" y1="68" x2="405" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="405" y="44" textAnchor="middle">LPT</text>
          <line x1="504" y1="78" x2="504" y2="50" stroke="#3A4A5A" strokeWidth="0.5" strokeDasharray="2,2"></line>
          <text x="504" y="44" textAnchor="middle">Core nozzle</text>
        </g>

        <line x1="280" y1="248" x2="280" y2="272" stroke="#2E3A48" strokeWidth="0.5" strokeDasharray="2,2"></line>
        <text x="280" y="284" textAnchor="middle" fill="#6B7C8E" fontSize="11" fontFamily="'IBM Plex Sans', sans-serif">Bypass duct — cold air</text>

        <text x="24" y="155" fill="#5C7C94" fontSize="17" textAnchor="middle">→</text>
        <text x="24" y="171" textAnchor="middle" fill="#6B7C8E" fontSize="9" fontFamily="'IBM Plex Mono', monospace">air</text>

        <g  fontFamily="'IBM Plex Mono', monospace" fontSize="7" fill="#5E7088" textAnchor="middle">
          <text x="70" y="236">s2</text>
          <text x="144" y="236">s24</text>
          <text x="220" y="236">s30</text>
          <text x="287" y="236">s40</text>
          <text x="337" y="236">s45</text>
          <text x="405" y="236">s50</text>
        </g>

        <text x="666" y="312" textAnchor="end" fill="#3E4C5C" fontSize="9" fontFamily="'IBM Plex Mono', monospace">generic representation · C-MAPSS FD001 class</text>

        <rect x="24" y="301" width="10" height="8" fill="#1E3A5C" rx="1"></rect>
        <text x="38" y="308" fill="#6B7C8E" fontSize="9" fontFamily="'IBM Plex Sans', sans-serif">cold section</text>
        <rect x="104" y="301" width="10" height="8" fill="#C03808" rx="1"></rect>
        <text x="118" y="308" fill="#6B7C8E" fontSize="9" fontFamily="'IBM Plex Sans', sans-serif">hot section</text>

        <g>
          {renderPaths()}
        </g>
      </svg>

        </div>
        
        <div className="min-h-[140px] border border-border rounded-lg bg-panel overflow-hidden">
          {panelContent}
        </div>
      </div>
    </PanelState>
  );
}
