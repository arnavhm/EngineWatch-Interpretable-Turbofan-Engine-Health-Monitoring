/**
 * src/components/EngineHealthMap.tsx  (REPLACE the existing file entirely)
 *
 * Proper turbofan cross-section — aerodynamic paths, compression stage lines,
 * bypass duct, core shaft. Same data contract as before; only the geometry changed.
 */

import { useState, useEffect, useRef } from "react";
import { getContributions } from "../api";
import type { ContributionsResponse, ModuleHeat } from "../types";

const C = {
  healthy:  "#2DD4A7",
  caution:  "#F5A524",
  warning:  "#FF5A5F",
  inactive: "#1E2A35",
  stroke:   "#2A3A4A",
  strokeDim:"#1B2735",
  bg:       "#0B1014",
  surface:  "#111820",
  textDim:  "#4A5A68",
  text:     "#9DB4C8",
  textBright:"#C8D9E8",
  core:     "#1E3448",
  shaft:    "#243444",
} as const;

function moduleColor(m: ModuleHeat): string {
  if (!m.is_active || m.direction === "inactive") return C.inactive;
  if (m.direction === "healthy") return C.healthy;
  return m.norm_magnitude >= 0.6 ? C.warning : C.caution;
}

function moduleOpacity(m: ModuleHeat): number {
  if (!m.is_active || m.direction === "inactive") return 1;
  return 0.20 + 0.80 * m.norm_magnitude;
}

function TooltipContent({ m }: { m: ModuleHeat }) {
  const color = moduleColor(m);
  if (!m.is_active) {
    return (
      <div style={{ padding: "10px 14px", maxWidth: 260 }}>
        <div style={{ fontWeight: 700, color: C.textBright, marginBottom: 4, fontSize: 12 }}>
          {m.display_name}
        </div>
        <div style={{ color: C.textDim, fontSize: 11 }}>
          No active sensors — flat under single operating condition
        </div>
      </div>
    );
  }
  const rank = m.norm_magnitude === 1.0 ? "dominant driver"
    : m.norm_magnitude >= 0.6 ? "major driver" : "minor driver";
  const dirLabel = m.direction === "healthy" ? "holding health" : "driving degradation";
  return (
    <div style={{ padding: "10px 14px", minWidth: 220, maxWidth: 290 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
        <div style={{ width: 8, height: 8, borderRadius: 2, background: color, flexShrink: 0 }} />
        <span style={{ fontWeight: 700, color: C.textBright, fontSize: 12 }}>{m.display_name}</span>
        <span style={{ color: C.textDim, fontSize: 10, marginLeft: "auto" }}>{rank}</span>
      </div>
      <div style={{ color: C.textDim, fontSize: 10, marginBottom: 8, borderBottom: `1px solid ${C.stroke}`, paddingBottom: 6 }}>
        {dirLabel}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
        {m.active_sensors.map((s) => (
          <div key={s.sensor_id} style={{ display: "flex", gap: 6, fontSize: 11 }}>
            <span style={{ color: C.textDim, width: 28, flexShrink: 0 }}>{s.sensor_id}</span>
            <span style={{ color: C.text, flex: 1 }}>{s.symbol}</span>
            <span style={{
              color: s.signed_contribution >= 0 ? C.healthy : C.warning,
              fontVariantNumeric: "tabular-nums",
            }}>
              {s.signed_contribution >= 0 ? "+" : ""}{s.signed_contribution.toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

interface Props {
  engineId: number;
  datasetId?: string;
}

export default function EngineHealthMap({ engineId, datasetId = "FD001" }: Props) {
  const [data, setData]       = useState<ContributionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [tip, setTip]         = useState({ x: 0, y: 0 });
  const svgRef                = useRef<SVGSVGElement>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getContributions(engineId, datasetId)
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [engineId, datasetId]);

  const heat = Object.fromEntries((data?.modules ?? []).map((m) => [m.module, m]));

  function fill(key: string)    { return heat[key] ? moduleColor(heat[key])   : C.inactive; }
  function opacity(key: string) { return heat[key] ? moduleOpacity(heat[key]) : 1; }

  function onEnter(key: string, e: React.MouseEvent<SVGElement>) {
    setHovered(key);
    updateTip(e);
  }
  function updateTip(e: React.MouseEvent<SVGElement>) {
    const r = svgRef.current?.getBoundingClientRect();
    if (r) setTip({ x: e.clientX - r.left + 14, y: e.clientY - r.top - 12 });
  }
  function onLeave() { setHovered(null); }

  const hoveredModule = hovered && heat[hovered] ? heat[hovered] : null;
  const dominant = data ? heat[data.dominant_module] : null;
  const dominantColor = dominant ? moduleColor(dominant) : C.textDim;

  // ── helper: module group ──────────────────────────────────────────────────
  function Mod({ id, children }: { id: string; children: React.ReactNode }) {
    return (
      <g
        onMouseEnter={(e) => onEnter(id, e)}
        onMouseMove={updateTip}
        onMouseLeave={onLeave}
        style={{ cursor: "pointer" }}
      >
        {children}
      </g>
    );
  }

  // ── stage lines (compression visual) ────────────────────────────────────
  function StageLines({ x1, y1, x2, y2, count = 5, clr }: {
    x1: number; y1: number; x2: number; y2: number; count?: number; clr: string;
  }) {
    const lines = [];
    for (let i = 1; i < count; i++) {
      const t = i / count;
      const lx = x1 + (x2 - x1) * t;
      const ly1 = y1 + (y2 - y1) * t * 0.3;
      const ly2 = y2 - (y2 - y1) * t * 0.3 + (y2 - y1) * t;
      lines.push(
        <line key={i} x1={lx} y1={Math.min(ly1, y1 + 4)} x2={lx} y2={Math.max(ly2, y2 - 4)}
          stroke={clr} strokeWidth="0.8" opacity="0.35" />
      );
    }
    return <>{lines}</>;
  }

  return (
    <div style={{
      background: C.bg,
      borderRadius: 12,
      padding: "18px 20px 14px",
      position: "relative",
      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
    }}>
      {/* header */}
      <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 14 }}>
        <span style={{ color: C.textBright, fontSize: 11, fontWeight: 700, letterSpacing: "0.08em" }}>
          ENGINE HEALTH MAP
        </span>
        <span style={{ color: C.textDim, fontSize: 10, letterSpacing: "0.04em" }}>
          PC1 ATTRIBUTION · {datasetId}
        </span>
        {loading && <span style={{ color: C.textDim, fontSize: 10, marginLeft: "auto" }}>loading…</span>}
        {error   && <span style={{ color: C.warning, fontSize: 10, marginLeft: "auto" }}>{error}</span>}
      </div>

      <div style={{ position: "relative" }}>
        <svg
          ref={svgRef}
          viewBox="0 0 780 240"
          style={{ width: "100%", display: "block" }}
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* ── BYPASS DUCT (top + bottom bands) ── */}
          <Mod id="bypass">
            {/* top bypass band */}
            <path
              d="M 158,52 L 620,44 L 640,56 L 620,56 L 158,64 Z"
              fill={fill("bypass")} opacity={opacity("bypass")}
              stroke={C.stroke} strokeWidth="0.8"
            />
            {/* bottom bypass band */}
            <path
              d="M 158,188 L 620,184 L 640,184 L 620,196 L 158,188 Z"
              fill={fill("bypass")} opacity={opacity("bypass")}
              stroke={C.stroke} strokeWidth="0.8"
            />
          </Mod>

          {/* ── NACELLE outer shell (top) ── */}
          <path
            d="M 72,120 Q 90,50 158,52 L 620,44 Q 665,42 695,80 L 710,120"
            fill="none" stroke={C.stroke} strokeWidth="1.5"
          />
          {/* nacelle outer shell (bottom) */}
          <path
            d="M 72,120 Q 90,190 158,188 L 620,196 Q 665,198 695,160 L 710,120"
            fill="none" stroke={C.stroke} strokeWidth="1.5"
          />
          {/* nozzle plug */}
          <path
            d="M 695,80 Q 730,100 740,120 Q 730,140 695,160"
            fill={C.core} stroke={C.stroke} strokeWidth="1"
          />

          {/* ── CORE inner cowl (top + bottom) ── */}
          <path d="M 158,74  L 620,68  L 640,80  L 620,80  L 158,86"
            fill="none" stroke={C.strokeDim} strokeWidth="0.8" />
          <path d="M 158,166 L 620,160 L 640,160 L 620,172 L 158,166"
            fill="none" stroke={C.strokeDim} strokeWidth="0.8" />

          {/* ── FAN ── */}
          <Mod id="fan">
            <path
              d="M 100,78 L 158,74 L 158,166 L 100,162 Q 72,155 72,120 Q 72,85 100,78 Z"
              fill={fill("fan")} opacity={opacity("fan")}
              stroke={C.stroke} strokeWidth="1"
            />
            {/* fan blade lines */}
            {[88,96,104,112,120,128,136,144,152].map((y) => (
              <line key={y} x1="74" y1={y} x2="157" y2={y + (y < 120 ? -2 : 2)}
                stroke={fill("fan")} strokeWidth="1.2" opacity="0.5" />
            ))}
            {/* nose cone */}
            <path d="M 72,120 Q 58,120 46,120" stroke={C.textDim} strokeWidth="1" fill="none"/>
            <ellipse cx="46" cy="120" rx="10" ry="14" fill={C.shaft} stroke={C.stroke} strokeWidth="1"/>
          </Mod>

          {/* ── LPC ── */}
          <Mod id="lpc">
            <path
              d="M 158,86 L 230,90 L 230,150 L 158,154 L 158,86 Z"
              fill={fill("lpc")} opacity={opacity("lpc")}
              stroke={C.stroke} strokeWidth="1"
            />
            <StageLines x1={158} y1={86} x2={230} y2={90} count={4} clr={fill("lpc")} />
          </Mod>

          {/* ── HPC — widest, most prominent ── */}
          <Mod id="hpc">
            <path
              d="M 230,90 L 420,96 L 420,144 L 230,150 Z"
              fill={fill("hpc")} opacity={opacity("hpc")}
              stroke={C.stroke} strokeWidth="1"
            />
            <StageLines x1={230} y1={90} x2={420} y2={96} count={9} clr={fill("hpc")} />
          </Mod>

          {/* ── BURNER / COMBUSTOR ── */}
          <Mod id="burner">
            <path
              d="M 420,96 L 490,98 L 490,142 L 420,144 Z"
              fill={fill("burner")} opacity={opacity("burner")}
              stroke={C.stroke} strokeWidth="1"
            />
            {/* combustor can suggestion */}
            <ellipse cx="455" cy="120" rx="22" ry="18"
              fill="none" stroke={fill("burner")} strokeWidth="1" opacity="0.4"/>
          </Mod>

          {/* ── HPT ── */}
          <Mod id="hpt">
            <path
              d="M 490,98 L 560,94 L 560,146 L 490,142 Z"
              fill={fill("hpt")} opacity={opacity("hpt")}
              stroke={C.stroke} strokeWidth="1"
            />
            <StageLines x1={490} y1={98} x2={560} y2={94} count={4} clr={fill("hpt")} />
          </Mod>

          {/* ── LPT ── */}
          <Mod id="lpt">
            <path
              d="M 560,94 L 648,80 L 648,160 L 560,146 Z"
              fill={fill("lpt")} opacity={opacity("lpt")}
              stroke={C.stroke} strokeWidth="1"
            />
            <StageLines x1={560} y1={94} x2={648} y2={80} count={6} clr={fill("lpt")} />
          </Mod>

          {/* ── CORE SHAFT ── */}
          <Mod id="core">
            <rect x="100" y="115" width="548" height="10" rx="3"
              fill={fill("core")} opacity={opacity("core")}
              stroke={C.stroke} strokeWidth="0.8"
            />
          </Mod>

          {/* ── EPR badge (off-engine) ── */}
          <Mod id="epr">
            <rect x="658" y="198" width="96" height="22" rx="11"
              fill={fill("epr")} opacity={0.7}
              stroke={C.stroke} strokeWidth="0.8"
            />
            <text x="706" y="212" textAnchor="middle"
              fill={C.textDim} fontSize="9" fontFamily="inherit">
              EPR · OVERALL
            </text>
          </Mod>

          {/* ── module labels ── */}
          {([
            { x: 115,  y: 213, label: "FAN",    key: "fan"    },
            { x: 194,  y: 213, label: "LPC",    key: "lpc"    },
            { x: 325,  y: 213, label: "HPC",    key: "hpc"    },
            { x: 455,  y: 213, label: "BURNER", key: "burner", dim: true },
            { x: 525,  y: 213, label: "HPT",    key: "hpt"    },
            { x: 604,  y: 213, label: "LPT",    key: "lpt"    },
          ] as Array<{ x: number; y: number; label: string; key: string; dim?: boolean }>)
            .map(({ x, y, label, key, dim }) => (
              <text key={label} x={x} y={y} textAnchor="middle"
                fill={hovered === key ? moduleColor(heat[key] ?? {} as ModuleHeat) : (dim ? C.textDim : C.text)}
                fontSize="10" fontWeight="600" letterSpacing="0.06em"
                fontFamily="inherit" style={{ transition: "fill 0.2s" }}>
                {label}
              </text>
            ))
          }
          <text x="390" y="38" textAnchor="middle"
            fill={C.textDim} fontSize="9" fontFamily="inherit">BYPASS DUCT</text>

          {/* ── flow direction arrow ── */}
          <defs>
            <marker id="arr" viewBox="0 0 8 8" refX="6" refY="4"
              markerWidth="5" markerHeight="5" orient="auto">
              <path d="M1 1L7 4L1 7" fill="none" stroke={C.textDim} strokeWidth="1.5"
                strokeLinecap="round" strokeLinejoin="round"/>
            </marker>
          </defs>
          <line x1="22" y1="120" x2="42" y2="120"
            stroke={C.textDim} strokeWidth="0.8" markerEnd="url(#arr)" />
          <text x="22" y="133" textAnchor="middle" fill={C.textDim} fontSize="8" fontFamily="inherit">
            airflow
          </text>

          {/* ── legend ── */}
          <g transform="translate(0,228)">
            {([
              { c: C.healthy,  l: "holding health"    },
              { c: C.caution,  l: "degrading"         },
              { c: C.warning,  l: "dominant driver"   },
              { c: C.inactive, l: "flat / inactive",  border: true },
            ] as Array<{ c: string; l: string; border?: boolean }>)
              .map(({ c, l, border }, i) => (
                <g key={l} transform={`translate(${i * 178}, 0)`}>
                  <rect width="10" height="10" rx="2" fill={c}
                    stroke={border ? C.stroke : "none"} strokeWidth="0.8" />
                  <text x="16" y="9" fill={C.textDim} fontSize="9" fontFamily="inherit">{l}</text>
                </g>
              ))
            }
          </g>
        </svg>

        {/* tooltip */}
        {hoveredModule && (
          <div style={{
            position: "absolute",
            left: Math.min(tip.x, 460),
            top: tip.y,
            background: "#0F1922",
            border: `1px solid ${C.stroke}`,
            borderRadius: 8,
            pointerEvents: "none",
            zIndex: 10,
            boxShadow: "0 8px 24px #00000080",
          }}>
            <TooltipContent m={hoveredModule} />
          </div>
        )}
      </div>

      {/* dominant driver strip */}
      {data && (
        <div style={{
          marginTop: 10,
          padding: "7px 12px",
          background: C.surface,
          borderRadius: 6,
          borderLeft: `3px solid ${dominantColor}`,
          display: "flex",
          alignItems: "center",
          gap: 8,
          flexWrap: "wrap",
        }}>
          <span style={{ color: C.textDim, fontSize: 10, letterSpacing: "0.05em" }}>
            DOMINANT DRIVER
          </span>
          <span style={{ color: C.textBright, fontSize: 11, fontWeight: 600 }}>
            {data.dominant_driver_text}
          </span>
          <span style={{ color: C.textDim, fontSize: 10, marginLeft: "auto" }}>
            relative attribution — absolute state on risk gauge
          </span>
        </div>
      )}

      {data?.modules.every((m) => !m.is_active) && (
        <div style={{ color: C.textDim, fontSize: 11, marginTop: 8 }}>
          No active sensor contributions for this cycle.
        </div>
      )}
    </div>
  );
}
