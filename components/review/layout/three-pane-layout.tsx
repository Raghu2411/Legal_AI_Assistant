import React from 'react';

interface ThreePaneLayoutProps {
  leftPane: React.ReactNode;
  centerPane: React.ReactNode;
  rightPane: React.ReactNode;
}

export function ThreePaneLayout({ leftPane, centerPane, rightPane }: ThreePaneLayoutProps) {
  return (
    <div className="flex h-[calc(100vh-4rem)] w-full overflow-hidden bg-background">
      {/* Left Pane: Risk List (25%) */}
      <aside className="w-1/4 border-r border-border h-full overflow-hidden flex flex-col">
        {leftPane}
      </aside>

      {/* Center Pane: Document Editor (50%) */}
      <main className="flex-1 border-r border-border h-full overflow-hidden flex flex-col bg-muted/30">
        {centerPane}
      </main>

      {/* Right Pane: Actions/Details (25%) */}
      <aside className="w-1/4 h-full overflow-hidden flex flex-col">
        {rightPane}
      </aside>
    </div>
  );
}
