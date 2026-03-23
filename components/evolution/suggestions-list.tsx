"use client";

import { useState, useEffect, useCallback } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Badge } from "@/components/ui/badge";
import { Loader2, ArrowRight, Check, X, RefreshCw } from "lucide-react";
import { toast } from "sonner";

interface SuggestionsListProps {
  standardId: string | null;
}

export function SuggestionsList({ standardId }: SuggestionsListProps) {
  const [suggestions, setSuggestions] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isApproving, setIsApproving] = useState(false);
  const [targetVersions, setTargetVersions] = useState<Record<string, number>>({});

  const fetchSuggestions = useCallback(async () => {
    if (!standardId) return;
    setIsLoading(true);
    try {
      const response = await fetch(`/api/evolution/suggestions/${standardId}`);
      if (!response.ok) throw new Error("Failed to fetch suggestions");
      const data = await response.json();
      setSuggestions(data.suggestions || []);
      
      // Initialize version tracking if targets were returned with versions
      // For this prototype, we assume the backend would provide them or we refresh them
    } catch (err) {
      toast.error("Could not load AI suggestions.");
    } finally {
      setIsLoading(false);
    }
  }, [standardId]);

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  };

  const handleApprove = async () => {
    if (selectedIds.length === 0) return;
    setIsApproving(true);
    try {
      const response = await fetch("/api/evolution/approve", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          suggestionIds: selectedIds,
          expectedVersions: targetVersions // T026 Concurrency Support
        }),
      });

      if (response.status === 409) {
        const errorData = await response.json();
        toast.error(errorData.error || "Concurrency Conflict detected.");
        return;
      }

      if (!response.ok) throw new Error("Approval failed");

      toast.success(`Successfully approved ${selectedIds.length} suggestions.`);
      setSelectedIds([]);
      fetchSuggestions(); // Refresh list
    } catch (err) {
      toast.error("Failed to approve selected suggestions.");
    } finally {
      setIsApproving(false);
    }
  };

  if (!standardId) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center space-y-4">
        <div className="p-4 rounded-full bg-muted">
          <RefreshCw className="h-10 w-10 text-muted-foreground" />
        </div>
        <div>
          <h3 className="text-lg font-semibold">No Standard Selected</h3>
          <p className="text-muted-foreground max-w-sm">
            Please upload or select a compliance standard from the Gap Analysis tab to view evolution suggestions.
          </p>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-lg">Retrieving AI suggestions...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold tracking-tight">Evolution Suggestions</h2>
          <p className="text-muted-foreground">
            Review AI-generated changes and approve them to evolve your firm&apos;s rules.
          </p>
        </div>
        <div className="flex space-x-2">
          <Button 
            variant="outline" 
            onClick={() => setSelectedIds(suggestions.map(s => s.id))}
            disabled={suggestions.length === 0}
          >
            Select All
          </Button>
          <Button 
            onClick={handleApprove} 
            disabled={selectedIds.length === 0 || isApproving}
          >
            {isApproving ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Check className="mr-2 h-4 w-4" />
            )}
            Approve Selected ({selectedIds.length})
          </Button>
        </div>
      </div>

      <div className="grid gap-6">
        {suggestions.map((s) => (
          <Card key={s.id} className={`overflow-hidden border-2 transition-all ${selectedIds.includes(s.id) ? "border-primary" : "border-transparent"}`}>
            <div className="bg-muted/50 px-4 py-2 border-b flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <Checkbox 
                  checked={selectedIds.includes(s.id)}
                  onCheckedChange={() => toggleSelection(s.id)}
                />
                <Badge variant={s.target_type === 'golden_rule' ? "default" : "secondary"}>
                  {s.target_type === 'golden_rule' ? "Golden Rule" : "Playbook Clause"}
                </Badge>
                <span className="text-xs text-muted-foreground font-mono">
                  {s.target_id === '00000000-0000-0000-0000-000000000000' ? "NEW" : s.target_id.slice(0, 8)}
                </span>
              </div>
              <Badge variant={s.status === 'pending' ? "outline" : "default"}>
                {s.status.toUpperCase()}
              </Badge>
            </div>
            <CardContent className="p-0">
              <div className="grid md:grid-cols-2 divide-x divide-border">
                <div className="p-4 space-y-2">
                  <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">Current Text</h4>
                  <div className="text-sm p-3 bg-red-50/50 rounded border border-red-100 line-through text-red-900/70">
                    {s.current_text || "None (New rule proposed)"}
                  </div>
                </div>
                <div className="p-4 space-y-2">
                  <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider">Suggested Text</h4>
                  <div className="text-sm p-3 bg-green-50/50 rounded border border-green-100 text-green-900">
                    {s.suggested_text}
                  </div>
                </div>
              </div>
              <div className="p-4 bg-muted/20 border-t">
                <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-1">AI Rationale</h4>
                <p className="text-sm italic">{s.rationale}</p>
              </div>
            </CardContent>
          </Card>
        ))}

        {suggestions.length === 0 && (
          <div className="text-center py-20 border-2 border-dashed rounded-xl">
            <X className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
            <h3 className="text-lg font-medium">No suggestions found</h3>
            <p className="text-muted-foreground">
              The AI auditor found no required changes for this standard chunk.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
