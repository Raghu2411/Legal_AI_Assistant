"use client";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { confirmObligation, rejectObligation } from "@/lib/operations/actions";
import { useState } from "react";
import { Calendar, Check, X, ShieldAlert, ShieldCheck, Eraser } from "lucide-react";
import { Input } from "@/components/ui/input";

export function ObligationItem({ obligation, onSuccess }: any) {
  const [loading, setLoading] = useState(false);
  const [date, setDate] = useState(obligation.due_date || "");

  const handleConfirm = async () => {
    if (!date) {
      alert("Due date is required for confirmation.");
      return;
    }
    setLoading(true);
    try {
      await confirmObligation(obligation.id, date);
      onSuccess();
    } catch (error) {
      alert("Failed to confirm obligation");
    } finally {
      setLoading(false);
    }
  };

  const handleReject = async () => {
    if (!confirm("Are you sure you want to REJECT this obligation? It will be removed from verification.")) return;
    setLoading(true);
    try {
      await rejectObligation(obligation.id);
      onSuccess();
    } catch (error) {
      alert("Failed to reject obligation");
    } finally {
      setLoading(false);
    }
  };

  const clearDate = () => setDate("");

  const compliance = (obligation.metadata?.compliance || []).filter(
    (c: any) => c.source !== "handbook"
  );

  return (
    <Card className="mb-4">
      <CardContent className="pt-6">
        <div className="flex items-start justify-between">
          <div className="space-y-2 flex-1 mr-4">
            <p className="text-sm font-medium leading-normal">{obligation.description}</p>
            <div className="flex flex-wrap gap-2">
              {compliance.map((c: any, i: number) => (
                <Badge key={i} variant={c.status === "passed" ? "outline" : "destructive"} className="text-[10px] py-0">
                  {c.status === "passed" ? (
                    <ShieldCheck className="mr-1 h-3 w-3" />
                  ) : (
                    <ShieldAlert className="mr-1 h-3 w-3" />
                  )}
                  {c.source === "admin" ? "Policy" : "Regulatory"}: {c.status.toUpperCase()}
                </Badge>
              ))}
              {!obligation.due_date && (
                <Badge variant="secondary" className="text-[10px] py-0">TBD DATE</Badge>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <div className="flex items-center gap-1 relative">
              <Calendar className="h-4 w-4 text-muted-foreground" />
              <div className="flex items-center gap-1 border rounded-md px-1 bg-background">
                <Input
                  type="date"
                  className="w-32 h-8 text-xs border-0 focus-visible:ring-0 p-0"
                  value={date ? (date.includes('T') ? date.split("T")[0] : date) : ""}
                  onChange={(e) => setDate(e.target.value)}
                />
                {date && (
                  <Button variant="ghost" size="icon" className="h-4 w-4 text-muted-foreground hover:text-foreground" onClick={clearDate} title="Clear Date">
                    <Eraser className="h-3 w-3" />
                  </Button>
                )}
              </div>
            </div>
            <Button size="icon" variant="outline" className="h-8 w-8 text-green-600 hover:text-green-700 hover:bg-green-50" onClick={handleConfirm} disabled={loading} title="Confirm & Add to Calendar">
              <Check className="h-4 w-4" />
            </Button>
            <Button size="icon" variant="outline" className="h-8 w-8 text-red-600 hover:text-red-700 hover:bg-red-50" onClick={handleReject} disabled={loading} title="Reject Obligation">
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
