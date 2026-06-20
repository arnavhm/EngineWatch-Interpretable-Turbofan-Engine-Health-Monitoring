/**
 * src/components/EngineHealthMap.tsx
 *
 * Live turbofan cross-section heat map driven by /predict/{engine}/contributions.
 * Each module is coloured by its PC1 attribution direction and scaled in
 * opacity by its relative magnitude (dominant module = opacity 1.0).
 *
 * EICAS colour convention (matches enginewatch.tech Tailwind tokens):
 *   Healthy      →  #2DD4A7  (holding HI high)
 *   Caution      →  #F5A524  (critical, moderate driver)
 *   Warning      →  #FF5A5F  (critical, dominant driver — norm_magnitude ≥ 0.6)
 *   Inactive     →  #2A333C  (no active sensor / flat under single condition)
 *
 * Colour gate:
 *   direction === "healthy"                         → green
 *   direction === "critical" && norm_magnitude ≥ 0.6 → red
 *   direction === "critical" && norm_magnitude < 0.6 → amber
 *   direction === "inactive"                        → dark grey, fixed opacity
 *
 * Opacity for active modules:  0.25 + 0.75 * norm_magnitude
 * Inactive modules: opacity 0.35, fixed.
 *
 * Verification target: Engine 34 / FD001 → HPC dominant, red, opacity 1.0.
 *
 * NO retraining. Inference only. This component is a pure consumer of the
 * /predict/{engine_id}/contributions endpoint.
 */

import { useState, useEffect, useRef } from "react";
import { getContributions } from "../api";
import type { ContributionsResponse, ModuleHeat } from "../types";

// ── EICAS tokens ─────────────────────────────────────────────────────────────
const C = {
  healthy:  "#2DD4A7",
  caution:  "#F5A524",
  warning:  "#FF5A5F",
  inactive: "#2A333C",
  stroke:   "#1B2430",
  bg:       "#0B1014",
  textDim:  "#6B7686",
  text:     "#C8D2DE",
  core:     "#3A5A80",
  axis:     "#6B768640",
} as const;

// ── colour + opacity helpers ──────────────────────────────────────────────────
function moduleColor(m: ModuleHeat): string {
  if (!m.is_active || m.direction === "inactive") return C.inactive;
  if (m.direction === "healthy") return C.healthy;
  return m.norm_magnitude >= 0.6 ? C.warning : C.caution;
}

function moduleOpacity(m: ModuleHeat): number {
  if (!m.is_active || m.direction === "inactive") return 0.35;
  return 0.25 + 0.75 * m.norm_magnitude;
}

// ── tooltip content ───────────────────────────────────────────────────────────
function TooltipContent({ m }: { m: ModuleHeat }) {
  if (!m.is_active) {
    return (
      <div style={{ padding: "10px 12px", maxWidth: 260 }}>
        <div style={{ fontWeight: 700, color: C.text, marginBottom: 4 }}>
          {m.display_name}
        </div>
        <div style={{ color: C.textDim, fontSize: 12 }}>
          No active sensors — flat under single operating condition
        </div>
      </div>
    );
  }

  const dirLabel =
    m.direction === "healthy" ? "holding health" : "driving degradation";
  const rank =
    m.norm_magnitude === 1.0
      ? "dominant PC1 driver"
      : m.norm_magnitude >= 0.6
      ? "major PC1 driver"
      : "minor PC1 driver";

  return (
    <div style={{ padding: "10px 12px", minWidth: 220, maxWidth: 300 }}>
      <div style={{ fontWeight: 700, color: C.text, marginBottom: 2 }}>
        {m.display_name}
      </div>
      <div style={{ color: C.textDim, fontSize: 11, marginBottom: 8 }}>
        {rank} · {dirLabel}
      </div>
      <table style={{ borderCollapse: "collapse", width: "100%", fontSize: 12 }}>
        <tbody>
          {m.active_sensors.map((s) => (
            <tr key={s.sensor_id}>
              <td style={{ color: C.textDim, paddingRight: 8, paddingBottom: 3 }}>
                {s.sensor_id}
              </td>
              <td style={{ color: C.text, paddingRight: 8, paddingBottom: 3 }}>
                {s.symbol}
              </td>
              <td
                style={{
                  color: s.signed_contribution >= 0 ? C.healthy : C.warning,
                  textAlign: "right",
                  paddingBottom: 3,
                  fontVariantNumeric: "tabular-nums",
                }}
              >
                {s.signed_contribution >= 0 ? "+" : ""}
                {s.signed_contribution.toFixed(2)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────
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

  // Build a lookup keyed by module string for O(1) access in SVG bindings
  const heat = Object.fromEntries(
    (data?.modules ?? []).map((m) => [m.module, m]),
  );

  function fill(key: string) {
    return heat[key] ? moduleColor(heat[key]) : C.inactive;
  }
  function opacity(key: string) {
    return heat[key] ? moduleOpacity(heat[key]) : 0.35;
  }

  function onEnter(key: string, e: React.MouseEvent<SVGElement>) {
    setHovered(key);
    const rect = svgRef.current?.getBoundingClientRect();
    if (rect) {
      setTip({ x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 8 });
    }
  }
  function onMove(e: React.MouseEvent<SVGElement>) {
    const rect = svgRef.current?.getBoundingClientRect();
    if (rect) {
      setTip({ x: e.clientX - rect.left + 12, y: e.clientY - rect.top - 8 });
    }
  }
  function onLeave() { setHovered(null); }

  const hoveredModule = hovered && heat[hovered] ? heat[hovered] : null;

  // ── module group factory (keeps JSX DRY) ────────────────────────────────────
  function Mod({
    id, children,
  }: { id: string; children: React.ReactNode }) {
    return (
      <g
        id={`module-${id}`}
        onMouseEnter={(e) => onEnter(id, e)}
        onMouseMove={onMove}
        onMouseLeave={onLeave}
        style={{ cursor: "pointer", transition: "opacity 0.25s ease" }}
      >
        {children}
      </g>
    );
  }

  return (
    <div
      style={{
        background: C.bg,
        borderRadius: 12,
        padding: "20px 24px",
        position: "relative",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
      }}
    >
      {/* ── header ── */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 16 }}>
        <span style={{ color: C.text, fontSize: 13, fontWeight: 700, letterSpacing: "0.06em" }}>
          ENGINE HEALTH MAP
        </span>
        <span style={{ color: C.textDim, fontSize: 11 }}>PC1 ATTRIBUTION BY MODULE</span>
        {loading && (
          <span style={{ color: C.textDim, fontSize: 11, marginLeft: "auto" }}>
            loading…
          </span>
        )}
        {error && (
          <span style={{ color: C.warning, fontSize: 11, marginLeft: "auto" }}>
            {error}
          </span>
        )}
      </div>

      {/* ── SVG engine cross-section ── */}
      <div style={{ position: "relative" }}>
        <svg
          ref={svgRef}
          viewBox="0 0 820 340"
          style={{ width: "100%", display: "block" }}
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* nacelle cowl */}
          <rect
            x="46" y="72" width="690" height="196"
            rx="72" ry="72"
            fill="none" stroke={C.stroke} strokeWidth="2"
          />
          {/* nozzle cone */}
          <polygon
            points="622,102 704,132 704,208 622,238"
            fill={C.core} opacity={0.4} stroke={C.stroke} strokeWidth="1"
          />
          {/* inlet lip */}
          <ellipse cx="62" cy="170" rx="14" ry="96" fill="none" stroke={C.stroke} strokeWidth="1.5" />
          {/* centerline */}
          <line x1="62" y1="170" x2="712" y2="170"
            stroke={C.axis} strokeWidth="1" strokeDasharray="4 5" />

          {/* ── BYPASS — top + bottom bands ── */}
          <Mod id="bypass">
            <polygon points="238,82 620,82 620,100 238,100"
              fill={fill("bypass")} opacity={opacity("bypass")}
              stroke={C.stroke} strokeWidth="1.5" />
            <polygon points="238,258 620,258 620,240 238,240"
              fill={fill("bypass")} opacity={opacity("bypass")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── FAN ── */}
          <Mod id="fan">
            <polygon points="96,78 168,90 168,250 96,262"
              fill={fill("fan")} opacity={opacity("fan")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── LPC ── */}
          <Mod id="lpc">
            <polygon points="168,102 238,110 238,230 168,238"
              fill={fill("lpc")} opacity={opacity("lpc")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── HPC ── */}
          <Mod id="hpc">
            <polygon points="238,110 372,134 372,206 238,230"
              fill={fill("hpc")} opacity={opacity("hpc")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── BURNER ── */}
          <Mod id="burner">
            <polygon points="372,134 432,134 432,206 372,206"
              fill={fill("burner")} opacity={opacity("burner")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── HPT ── */}
          <Mod id="hpt">
            <polygon points="432,132 512,130 512,210 432,208"
              fill={fill("hpt")} opacity={opacity("hpt")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── LPT ── */}
          <Mod id="lpt">
            <polygon points="512,130 622,102 622,238 512,210"
              fill={fill("lpt")} opacity={opacity("lpt")}
              stroke={C.stroke} strokeWidth="1.5" />
          </Mod>

          {/* ── CORE SHAFT — sits on top ── */}
          <Mod id="core">
            <rect x="110" y="163" width="512" height="14" rx="4"
              fill={fill("core")} opacity={opacity("core")}
              stroke={C.stroke} strokeWidth="1" />
          </Mod>

          {/* ── EPR / OVERALL ── */}
          <Mod id="epr">
            <rect x="640" y="272" width="128" height="26" rx="13"
              fill={fill("epr")} opacity={opacity("epr")}
              stroke={C.stroke} strokeWidth="1.5" />
            <text x="704" y="289" textAnchor="middle"
              fill={C.textDim} fontSize="10" fontFamily="inherit">
              EPR · OVERALL
            </text>
          </Mod>

          {/* ── labels ── */}
          {(
            [
              { x: 132, label: "FAN" },
              { x: 203, label: "LPC" },
              { x: 305, label: "HPC" },
              { x: 402, label: "BURNER", dim: true },
              { x: 472, label: "HPT" },
              { x: 567, label: "LPT" },
            ] as Array<{ x: number; label: string; dim?: boolean }>
          ).map(({ x, label, dim }) => (
            <text key={label} x={x} y={292} textAnchor="middle"
              fill={dim ? C.textDim : C.text}
              fontSize="11" fontWeight="600" letterSpacing="0.04em"
              fontFamily="inherit" pointerEvents="none">
              {label}
            </text>
          ))}
          <text x="140" y="76" textAnchor="middle"
            fill={C.textDim} fontSize="10" fontFamily="inherit" pointerEvents="none">
            BYPASS
          </text>

          {/* ── legend ── */}
          <g transform="translate(0,316)">
            {(
              [
                { color: C.healthy,  label: "holding health" },
                { color: C.caution,  label: "driving degradation" },
                { color: C.warning,  label: "dominant driver" },
                { color: C.inactive, label: "inactive (flat sensor)" },
              ] as Array<{ color: string; label: string }>
            ).map(({ color, label }, i) => (
              <g key={label} transform={`translate(${i * 190}, 0)`}>
                <rect width="12" height="12" rx="2" fill={color} />
                <text x="18" y="10" fill={C.textDim} fontSize="10" fontFamily="inherit">
                  {label}
                </text>
              </g>
            ))}
          </g>
        </svg>

        {/* ── hover tooltip ── */}
        {hoveredModule && (
          <div
            style={{
              position: "absolute",
              left: tip.x,
              top: tip.y,
              background: "#141C24",
              border: `1px solid ${C.stroke}`,
              borderRadius: 8,
              pointerEvents: "none",
              zIndex: 10,
              boxShadow: "0 4px 20px #00000060",
            }}
          >
            <TooltipContent m={hoveredModule} />
          </div>
        )}
      </div>

      {/* ── dominant driver one-liner ── */}
      {data && (
        <div
          style={{
            marginTop: 14,
            padding: "8px 12px",
            background: "#141C24",
            borderRadius: 6,
            borderLeft: `3px solid ${
              heat[data.dominant_module]
                ? moduleColor(heat[data.dominant_module])
                : C.inactive
            }`,
          }}
        >
          <span style={{ color: C.textDim, fontSize: 11 }}>DOMINANT DRIVER&nbsp;&nbsp;</span>
          <span style={{ color: C.text, fontSize: 12, fontWeight: 600 }}>
            {data.dominant_driver_text}
          </span>
          <span style={{ color: C.textDim, fontSize: 11, marginLeft: 8 }}>
            · relative attribution — absolute state on risk gauge
          </span>
        </div>
      )}

      {/* ── empty / all-inactive guard ── */}
      {data && data.modules.every((m) => !m.is_active) && (
        <div style={{ color: C.textDim, fontSize: 12, marginTop: 8 }}>
          No active sensor contributions for this cycle.
        </div>
      )}
    </div>
  );
}
