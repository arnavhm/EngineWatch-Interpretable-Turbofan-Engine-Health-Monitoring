import { useMemo } from 'react';
import { useContributions } from '../hooks/useContributions';

interface Props {
  engineId: number;
  datasetId: string;
}

function directionStyle(direction: string) {
  if (direction === 'healthy') {
    return {
      bar: 'bg-healthy',
      text: 'text-healthy',
      bannerBg: 'bg-healthy/10',
      bannerBorder: 'border-healthy/30',
    };
  }
  if (direction === 'critical') {
    return {
      bar: 'bg-critical',
      text: 'text-critical',
      bannerBg: 'bg-critical/10',
      bannerBorder: 'border-critical/30',
    };
  }
  return {
    bar: 'bg-faint',
    text: 'text-muted',
    bannerBg: 'bg-panel2',
    bannerBorder: 'border-border',
  };
}

function LoadingSkeleton() {
  return (
    <div className="flex flex-col gap-4 animate-pulse">
      <div className="h-10 bg-panel2 rounded-lg" />
      <div className="flex flex-col gap-2 px-1">
        {[1, 0.65, 0.4].map((w, i) => (
          <div key={i} className="flex items-center gap-3">
            <div className="w-10 h-2.5 bg-panel2 rounded" />
            <div className="h-2.5 bg-panel2 rounded" style={{ width: `${w * 55}%` }} />
          </div>
        ))}
      </div>
      <div className="h-28 bg-panel2 rounded-lg" />
    </div>
  );
}

export default function SensorExplanationPanel({ engineId, datasetId }: Props) {
  const { data, loading, error } = useContributions(engineId, datasetId);

  const activeModules = useMemo(
    () =>
      data
        ? [...data.modules]
            .filter(m => m.is_active)
            .sort((a, b) => b.norm_magnitude - a.norm_magnitude)
        : [],
    [data],
  );

  const dominantModule = useMemo(
    () => (data ? (data.modules.find(m => m.module === data.dominant_module) ?? null) : null),
    [data],
  );

  const dominantSensors = useMemo(
    () =>
      dominantModule
        ? [...dominantModule.active_sensors].sort(
            (a, b) => b.abs_contribution - a.abs_contribution,
          )
        : [],
    [dominantModule],
  );

  const maxAbsContrib = useMemo(
    () =>
      dominantSensors.length > 0
        ? Math.max(...dominantSensors.map(s => s.abs_contribution))
        : 1,
    [dominantSensors],
  );

  if (loading) return <LoadingSkeleton />;

  if (error) return <p className="text-muted text-sm">Sensor data unavailable</p>;

  if (!data || activeModules.length === 0) {
    return (
      <p className="text-muted text-sm">
        No active sensor contributions detected for this engine.
      </p>
    );
  }

  const domStyle = directionStyle(dominantModule?.direction ?? 'inactive');

  return (
    <div className="flex flex-col gap-5">
      {/* 1 — Dominant Driver Banner */}
      <div className={`rounded-lg border px-4 py-3 ${domStyle.bannerBg} ${domStyle.bannerBorder}`}>
        <p className={`text-sm font-semibold font-mono leading-snug ${domStyle.text}`}>
          {data.dominant_driver_text}
        </p>
      </div>

      {/* 2 — Module Heat Bars */}
      <div className="flex flex-col gap-2">
        {activeModules.map(mod => {
          const st = directionStyle(mod.direction);
          return (
            <div key={mod.module} className="flex items-center gap-3">
              <span className="w-10 shrink-0 text-right text-xs font-mono text-muted">
                {mod.display_name}
              </span>
              <div className="flex-1 h-2 rounded-full bg-panel2 overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-300 ${st.bar}`}
                  style={{ width: `${mod.norm_magnitude * 100}%` }}
                />
              </div>
              <span className="w-8 shrink-0 text-right text-[11px] font-mono text-faint">
                {(mod.norm_magnitude * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>

      {/* 3 — Sensor Contribution Table (dominant module) */}
      {dominantSensors.length > 0 && (
        <div className="rounded-lg border border-border overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-border bg-panel2">
                <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-faint font-normal w-20">
                  Symbol
                </th>
                <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wide text-faint font-normal">
                  Description
                </th>
                <th className="px-3 py-2 text-center text-[10px] uppercase tracking-wide text-faint font-normal w-36">
                  Contribution
                </th>
              </tr>
            </thead>
            <tbody>
              {dominantSensors.map(sensor => {
                const barPct = maxAbsContrib > 0
                  ? (sensor.abs_contribution / maxAbsContrib) * 46
                  : 0;
                const isDegrading = sensor.signed_contribution > 0;
                return (
                  <tr
                    key={sensor.sensor_id}
                    className="border-b border-border/50 last:border-0 hover:bg-panel transition-colors"
                  >
                    <td className="px-3 py-2.5 w-20">
                      <div className="flex flex-col gap-0.5">
                        <span className="font-mono text-text text-xs">{sensor.symbol}</span>
                        <span className="font-mono text-[10px] text-faint">{sensor.sensor_id}</span>
                      </div>
                    </td>
                    <td className="px-3 py-2.5">
                      <span className="text-muted text-[11px] leading-snug">{sensor.description}</span>
                    </td>
                    <td className="px-3 py-2.5 w-36">
                      <div className="relative h-3 w-full">
                        <div className="absolute left-1/2 top-0 w-px h-full bg-border" />
                        {isDegrading ? (
                          <div
                            className="absolute top-0.5 h-2 rounded-sm bg-critical"
                            style={{ left: '50%', width: `${barPct}%` }}
                          />
                        ) : (
                          <div
                            className="absolute top-0.5 h-2 rounded-sm bg-accent"
                            style={{ right: '50%', width: `${barPct}%` }}
                          />
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
