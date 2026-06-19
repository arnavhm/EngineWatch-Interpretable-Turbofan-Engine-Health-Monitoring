import type { RiskState } from './types';

export const STATE_TEXT: Record<RiskState, string> = {
  Healthy: "text-healthy",
  Degrading: "text-degrading",
  Critical: "text-critical"
};

export const STATE_VAR: Record<RiskState, string> = {
  Healthy: "var(--color-healthy)",
  Degrading: "var(--color-degrading)",
  Critical: "var(--color-critical)"
};
