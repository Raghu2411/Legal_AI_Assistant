import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";

interface RiskListPaneProps {
  children: React.ReactNode;
  searchTerm: string;
  onSearchChange: (value: string) => void;
  statusFilter: string;
  onStatusFilterChange: (value: string) => void;
}

export function RiskListPane({ 
  children, 
  searchTerm, 
  onSearchChange, 
  statusFilter, 
  onStatusFilterChange 
}: RiskListPaneProps) {
  return (
    <div className="flex flex-col h-full bg-muted/10">
      <div className="p-4 border-b border-border bg-background space-y-4">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Contract Risks</h2>
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input 
            placeholder="Search risks..." 
            className="pl-8 bg-muted/50 border-none h-9"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>
        <Tabs value={statusFilter} onValueChange={onStatusFilterChange} className="w-full">
          <TabsList className="grid w-full grid-cols-4 h-8 bg-muted">
            <TabsTrigger value="all" className="text-xs">All</TabsTrigger>
            <TabsTrigger value="red" className="text-xs text-red-600">Red</TabsTrigger>
            <TabsTrigger value="yellow" className="text-xs text-yellow-600">Yel</TabsTrigger>
            <TabsTrigger value="green" className="text-xs text-green-600">Grn</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-3">
          {children}
        </div>
      </ScrollArea>
    </div>
  );
}
