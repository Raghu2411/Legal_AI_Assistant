"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AuditPanel } from "./audit-panel";
import { SuggestionsList } from "./suggestions-list";
import { HistoryView } from "./history-view";

interface EvolutionStudioDashboardProps {
  initialStandards: any[];
  playbooks: any[];
  goldenRules: any[];
}

export function EvolutionStudioDashboard({
  initialStandards,
  playbooks,
  goldenRules,
}: EvolutionStudioDashboardProps) {
  const [activeStandardId, setActiveStandardId] = useState<string | null>(
    initialStandards[0]?.id || null
  );

  return (
    <Tabs defaultValue="audit" className="space-y-4">
      <TabsList>
        <TabsTrigger value="audit">Gap Analysis</TabsTrigger>
        <TabsTrigger value="review">Review Suggestions</TabsTrigger>
        <TabsTrigger value="history">Audit Trail</TabsTrigger>
      </TabsList>

      <TabsContent value="audit" className="space-y-4">
        <AuditPanel 
          onStandardSelect={setActiveStandardId}
          selectedStandardId={activeStandardId}
          initialStandards={initialStandards}
          playbooks={playbooks}
        />
      </TabsContent>

      <TabsContent value="review">
        <SuggestionsList standardId={activeStandardId} />
      </TabsContent>

      <TabsContent value="history">
        <HistoryView />
      </TabsContent>
    </Tabs>
  );
}
