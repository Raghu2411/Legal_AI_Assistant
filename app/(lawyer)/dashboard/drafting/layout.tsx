import React from 'react';

export default function DraftingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-[calc(100vh-4rem)] flex-col overflow-hidden bg-background">
      <div className="flex-1 overflow-hidden">
        {children}
      </div>
    </div>
  );
}
