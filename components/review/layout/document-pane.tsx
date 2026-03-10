import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";

interface DocumentPaneProps {
  title: string;
  children: React.ReactNode;
}

export function DocumentPane({ title, children }: DocumentPaneProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border bg-background">
        <h2 className="text-sm font-semibold truncate">{title}</h2>
      </div>
      <ScrollArea className="flex-1 p-6">
        <Card className="max-w-4xl mx-auto min-h-[calc(100vh-12rem)] p-12 shadow-sm bg-white">
          {children}
        </Card>
      </ScrollArea>
    </div>
  );
}
