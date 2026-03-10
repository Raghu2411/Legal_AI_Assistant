import React from 'react';
import { ScrollArea } from "@/components/ui/scroll-area";
import { ClauseAnalysis } from "@/lib/review/schemas";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Info, MessageSquare, Check, X, ShieldAlert } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";

interface ActionPaneProps {
  selectedRisk: ClauseAnalysis | null;
  onAcceptSuggestion: () => void;
  onOverrideStatus: (status: 'green' | 'yellow' | 'red', rationale: string) => void;
}

export function ActionPane({ selectedRisk, onAcceptSuggestion, onOverrideStatus }: ActionPaneProps) {
  const [overrideRationale, setOverrideRationale] = React.useState("");

  if (!selectedRisk) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center p-8 space-y-4 opacity-50 bg-muted/5">
        <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
          <Info className="h-6 w-6" />
        </div>
        <div>
          <p className="text-sm font-semibold">No Clause Selected</p>
          <p className="text-xs text-muted-foreground">Select a risk from the list to take action.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full border-l border-border bg-background">
      <div className="p-4 border-b border-border bg-background">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Action Studio</h2>
      </div>
      
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-6">
          {/* AI Rationale Section */}
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4 text-muted-foreground" />
              <h3 className="text-xs font-bold uppercase tracking-tight">AI Reasoning</h3>
            </div>
            <Card className="p-4 bg-muted/30 border-none">
              <p className="text-sm leading-relaxed italic">&quot;{selectedRisk.rationale}&quot;</p>
            </Card>
          </section>

          {/* Suggestion Section */}
          {selectedRisk.suggested_rewrite && (
            <section className="space-y-3">
              <div className="flex items-center gap-2">
                <MessageSquare className="h-4 w-4 text-primary" />
                <h3 className="text-xs font-bold uppercase tracking-tight text-primary">Proposed Redline</h3>
              </div>
              <Card className="p-4 border-primary/20 bg-primary/5">
                <p className="text-sm leading-relaxed mb-4">{selectedRisk.suggested_rewrite}</p>
                <Button className="w-full" size="sm" onClick={onAcceptSuggestion}>
                  <Check className="h-3 w-3 mr-2" />
                  Apply Suggestion
                </Button>
              </Card>
            </section>
          )}

          {/* Manual Override Section */}
          <section className="space-y-3 pt-6 border-t border-border">
            <h3 className="text-xs font-bold uppercase tracking-tight text-muted-foreground">Manual Override</h3>
            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-2">
                <Button variant="outline" size="sm" className="h-8 border-red-200 text-red-600 hover:bg-red-50" onClick={() => onOverrideStatus('red', overrideRationale)}>RED</Button>
                <Button variant="outline" size="sm" className="h-8 border-yellow-200 text-yellow-600 hover:bg-yellow-50" onClick={() => onOverrideStatus('yellow', overrideRationale)}>YEL</Button>
                <Button variant="outline" size="sm" className="h-8 border-green-200 text-green-600 hover:bg-green-50" onClick={() => onOverrideStatus('green', overrideRationale)}>GRN</Button>
              </div>
              <Textarea 
                placeholder="Rationale for override (mandatory)..." 
                className="text-xs min-h-[100px] resize-none"
                value={overrideRationale}
                onChange={(e) => setOverrideRationale(e.target.value)}
              />
            </div>
          </section>
        </div>
      </ScrollArea>
    </div>
  );
}
