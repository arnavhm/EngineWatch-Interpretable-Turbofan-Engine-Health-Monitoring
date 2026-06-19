import React from 'react';

interface PanelStateProps {
  loading?: boolean;
  error?: string | null;
  children: React.ReactNode;
}

export default function PanelState({ loading, error, children }: PanelStateProps) {
  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[150px] bg-panel2 bg-opacity-50 border border-dashed border-border rounded-lg">
        <span className="text-muted text-sm border-l-2 border-critical pl-4 py-2">Error: {error}</span>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center min-h-[150px] animate-pulse">
        <div className="w-full h-full bg-panel2 rounded-lg border border-border min-h-[150px]"></div>
      </div>
    );
  }

  return <>{children}</>;
}
