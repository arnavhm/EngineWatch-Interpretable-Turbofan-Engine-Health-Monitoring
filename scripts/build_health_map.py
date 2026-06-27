import re

with open("frontend/src/components/raw_svg.txt") as f:
    svg = f.read()

# Convert attributes to camelCase
svg = svg.replace('stop-color', 'stopColor')
svg = svg.replace('stroke-width', 'strokeWidth')
svg = svg.replace('stroke-dasharray', 'strokeDasharray')
svg = svg.replace('stroke-linecap', 'strokeLinecap')
svg = svg.replace('font-family', 'fontFamily')
svg = svg.replace('font-size', 'fontSize')
svg = svg.replace('font-weight', 'fontWeight')
svg = svg.replace('text-anchor', 'textAnchor')

# Remove Claude design bindings
svg = re.sub(r'style="\{\{[^}]+\}\}"', '', svg)
svg = re.sub(r'onMouseOver="\{\{[^}]+\}\}"', '', svg)
svg = re.sub(r'onMouseLeave="\{\{[^}]+\}\}"', '', svg)

# Strip out the paths inside the interactive g and we will generate them manually
# We will just inject `{renderPaths()}` inside the `<g>` at the end
svg = re.sub(r'<g>\s*<path data-id="fan".*?</g>', '<g>{renderPaths()}</g>', svg, flags=re.DOTALL)
# The `<g>` tag itself had bindings, we need to match it properly
svg = re.sub(r'<g onMouseOver.*?>(.*?)</g>', '<g>{renderPaths()}</g>', svg, flags=re.DOTALL)

component_code = f"""import React, {{ useState }} from 'react';
import {{ useContributions }} from '../hooks/useContributions';
import PanelState from './PanelState';

interface EngineHealthMapProps {{
  engineId: number;
  datasetId: string;
}}

const STATIC_BLURBS: Record<string, any> = {{
  fan: {{ name: 'Fan', spool: 'N1 · LP spool', blurb: 'About eleven wide-chord blades on the low-pressure spool. The air it drives through the bypass duct supplies most of the engine’s thrust.' }},
  lpc: {{ name: 'LPC', spool: 'N1 · LP spool', blurb: 'Five axial stages (the booster) that pre-compress core flow ahead of the HPC. Shares the LP spool with the fan and LPT.' }},
  hpc: {{ name: 'HPC', spool: 'N2 · HP spool', blurb: 'Eight dense stages delivering the bulk of the pressure rise. Interstage bleed ports tap air for turbine cooling and cabin systems.' }},
  comb: {{ name: 'Combustor', spool: 'Static structure', blurb: 'Fuel is injected and burned here — the hottest gas in the engine and the cold-to-hot transition point of the cycle.' }},
  hpt: {{ name: 'HPT', spool: 'N2 · HP spool', blurb: 'Three stages that extract work to drive the HPC. Blades are film-cooled with bleed air ducted forward from the compressor.' }},
  lpt: {{ name: 'LPT', spool: 'N1 · LP spool', blurb: 'Five progressively larger stages driving the fan and LPC through the LP spool, which runs the full length of the core.' }},
  nozzle: {{ name: 'Core nozzle', spool: 'Static structure', blurb: 'Converging nozzle and tail cone that accelerate the spent core gas, adding residual core thrust to the bypass stream.' }},
  bypass: {{ name: 'Bypass duct', spool: 'Cold flow path', blurb: 'The annular channel between nacelle and core carries most of the mass flow around the engine — the dominant thrust source.' }}
}};

const SVG_PATHS: Record<string, string> = {{
  fan: "M45,88 L108,72 L108,248 L45,232 Z",
  lpc: "M108,100 L180,100 L180,220 L108,220 Z",
  hpc: "M180,100 L260,100 L260,220 L180,220 Z",
  comb: "M260,100 L315,103 L315,217 L260,220 Z",
  hpt: "M315,103 L360,105 L360,215 L315,217 Z",
  lpt: "M360,105 L450,110 L450,210 L360,215 Z",
  nozzle: "M450,110 L558,132 L566,142 L566,178 L558,188 L450,210 Z",
  bypass: "M108,78 L440,86 L510,104 L546,124 L558,133 L450,110 L360,105 L315,103 L260,100 L180,100 L108,100 Z M108,242 L440,234 L510,216 L546,196 L558,187 L450,210 L360,215 L315,217 L260,220 L180,220 L108,220 Z"
}};

const API_MAP: Record<string, string> = {{
  fan: 'fan',
  lpc: 'lpc',
  hpc: 'hpc',
  comb: 'hpc',
  hpt: 'hpt',
  lpt: 'lpt',
  nozzle: '',
  bypass: 'bypass'
}};

export default function EngineHealthMap({{ engineId, datasetId }}: EngineHealthMapProps) {{
  const {{ data, loading, error }} = useContributions(engineId, datasetId);
  const [hoveredZone, setHoveredZone] = useState<string | null>(null);

  const getZoneColor = (zone: string, isHovered: boolean) => {{
    if (!data) return {{ fill: 'transparent', stroke: 'transparent' }};
    const apiKey = API_MAP[zone];
    
    // Base colors
    let r = 111, g = 168, b = 216; // default blueish
    let alpha = 0;
    let stroke = 'transparent';
    
    if (apiKey) {{
      const mod = data.modules.find(m => m.module.toLowerCase() === apiKey.toLowerCase());
      if (mod) {{
        // Color by direction
        if (mod.direction === 'critical') {{
          r = 224; g = 83; b = 58; // #E0533A
        }} else if (mod.direction === 'healthy') {{
          r = 62; g = 207; b = 142; // #3ECF8E
        }} else {{
          r = 224; g = 169; b = 58; // #E0A93A degrading
        }}
        
        alpha = mod.norm_magnitude * 0.6; // Scale opacity by severity
        
        if (data.dominant_module.toLowerCase() === apiKey.toLowerCase()) {{
          stroke = `rgba(${{r}},${{g}},${{b}}, 1)`;
          alpha = Math.max(alpha, 0.3); // Ensure visible
        }}
      }}
    }}
    
    if (isHovered) {{
      alpha = Math.min(alpha + 0.2, 0.8);
      stroke = stroke === 'transparent' ? 'rgba(255,255,255,0.3)' : stroke;
    }}
    
    return {{
      fill: `rgba(${{r}},${{g}},${{b}},${{alpha}})`,
      stroke,
      strokeWidth: 1.4,
      cursor: 'pointer',
      transition: 'fill .15s ease, stroke .15s ease'
    }};
  }};

  const renderPaths = () => {{
    return Object.entries(SVG_PATHS).map(([zone, pathData]) => (
      <path
        key={{zone}}
        d={{pathData}}
        style={{getZoneColor(zone, hoveredZone === zone)}}
        onMouseEnter={{() => setHoveredZone(zone)}}
        onMouseLeave={{() => setHoveredZone(null)}}
      />
    ));
  }};

  // Find info for the details panel
  let panelContent = null;
  if (hoveredZone) {{
    const staticInfo = STATIC_BLURBS[hoveredZone];
    const apiKey = API_MAP[hoveredZone];
    const mod = apiKey && data ? data.modules.find(m => m.module.toLowerCase() === apiKey.toLowerCase()) : null;
    
    let isDominant = false;
    if (mod && data) {{
       isDominant = data.dominant_module.toLowerCase() === mod.module.toLowerCase();
    }}
    
    panelContent = (
      <div className="flex flex-col gap-2 p-4 h-full bg-panel2/30">
        <div className="flex items-center gap-3">
          <span className="text-xl font-bold text-accent">{{mod ? mod.display_name : staticInfo.name}}</span>
          <span className="text-xs text-muted font-mono bg-panel border border-border px-2 py-1 rounded">{{staticInfo.spool}}</span>
          {{mod && (
            <span className={{`text-xs font-bold px-2 py-1 rounded ${{mod.direction === 'critical' ? 'bg-[#E0533A]/20 text-[#E0533A]' : (mod.direction === 'healthy' ? 'bg-[#3ECF8E]/20 text-[#3ECF8E]' : 'bg-[#E0A93A]/20 text-[#E0A93A]')}}`}}>
              {{mod.direction.toUpperCase()}}
            </span>
          )}}
        </div>
        
        {{isDominant && data && (
          <div className="text-xs text-[#E0533A] font-medium border-l-2 border-[#E0533A] pl-2 mb-2">
            {{data.dominant_driver_text}}
          </div>
        )}}
        
        {{mod && mod.active_sensors && mod.active_sensors.length > 0 ? (
          <div className="mt-2">
            <div className="text-xs text-muted mb-1 font-bold uppercase tracking-wider">Active Sensors</div>
            <div className="grid grid-cols-1 gap-1">
              {{mod.active_sensors.map(s => (
                <div key={{s.symbol}} className="flex items-center justify-between text-sm bg-panel border border-border p-1.5 rounded">
                  <div className="flex items-center gap-2">
                    <span className="font-mono font-bold">{{s.symbol}}</span>
                    <span className="text-muted text-xs truncate max-w-[150px]">{{s.description}}</span>
                  </div>
                  <span className={{`font-mono font-bold ${{s.signed_contribution < 0 ? 'text-[#E0533A]' : 'text-[#3ECF8E]'}}`}}>
                    {{s.signed_contribution > 0 ? '+' : ''}}{{s.signed_contribution.toFixed(2)}}
                  </span>
                </div>
              ))}}
            </div>
          </div>
        ) : (
          <div className="text-sm text-muted mt-2 border-l-2 border-border pl-3">
            {{staticInfo.blurb}}
          </div>
        )}}
      </div>
    );
  }} else {{
    panelContent = (
      <div className="flex items-center justify-center h-full text-muted text-sm font-mono opacity-50 p-6 text-center">
        Hover a section of the engine to inspect its live health contribution.
      </div>
    );
  }}

  return (
    <PanelState loading={{loading}} error={{error}}>
      <div className="w-full flex flex-col gap-4">
        <div className="w-full rounded-lg bg-[#080E18] overflow-hidden border border-[#16202E]">
          {svg}
        </div>
        
        <div className="min-h-[140px] border border-border rounded-lg bg-panel overflow-hidden">
          {{panelContent}}
        </div>
      </div>
    </PanelState>
  );
}}
"""

with open("frontend/src/components/EngineHealthMap.tsx", "w") as f:
    f.write(component_code)
