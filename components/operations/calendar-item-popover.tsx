"use client";

import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { revertObligationToPending, deleteObligation } from "@/lib/operations/actions";
import { useState } from "react";
import { ShieldAlert, ShieldCheck, Trash2, Undo2, User, FileText, Fingerprint } from "lucide-react";

export function CalendarItemPopover({ obligation, onClose, onSuccess }: any) {
  const [loading, setLoading] = useState(false);

  const handleRevert = async () => {
    if (!confirm("Revert this obligation to 'Pending'? it will move back to the Verification list.")) return;
    setLoading(true);
    try {
      await revertObligationToPending(obligation.id);
      onSuccess?.();
      onClose();
    } catch (error) {
      alert("Failed to revert obligation");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm("Are you sure you want to PERMANENTLY DELETE this obligation?")) return;
    setLoading(true);
    try {
      await deleteObligation(obligation.id);
      onSuccess?.();
      onClose();
    } catch (error) {
      alert("Failed to delete obligation");
    } finally {
      setLoading(false);
    }
  };

  const compliance = obligation.metadata?.compliance || [];

  return (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Obligation Details</DialogTitle>
        </DialogHeader>
        <div className="space-y-4 py-4">
          <div className="grid grid-cols-2 gap-4 bg-muted/30 p-3 rounded-lg border">
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground uppercase">
                <User className="h-3 w-3" />
                Client
              </div>
              <p className="text-sm font-medium">{obligation.clients?.name || "Unknown"}</p>
            </div>
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground uppercase">
                <Fingerprint className="h-3 w-3" />
                Client ID
              </div>
              <p className="text-sm font-mono">{obligation.clients?.auto_case_id || "N/A"}</p>
            </div>
            <div className="col-span-2 space-y-1 pt-2 border-t">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground uppercase">
                <FileText className="h-3 w-3" />
                Document
              </div>
              <p className="text-sm italic">{obligation.documents?.file_name || "N/A"}</p>
            </div>
          </div>

          <div>
            <h4 className="text-sm font-semibold mb-1 uppercase text-xs text-muted-foreground">Description</h4>
            <p className="text-sm leading-relaxed">{obligation.description}</p>
          </div>

          <div className="flex justify-between items-center border-t pt-4">
            <div>
              <h4 className="text-xs font-semibold mb-1 uppercase text-muted-foreground">Due Date</h4>
              <p className="text-sm font-medium">{new Date(obligation.due_date).toLocaleDateString(undefined, { dateStyle: 'long' })}</p>
            </div>
          </div>

          <div>
            <h4 className="text-xs font-semibold mb-2 uppercase text-muted-foreground">Compliance Status</h4>
            <div className="flex flex-wrap gap-2">
              {compliance.filter((c: any) => c.source !== "handbook").map((c: any, i: number) => (
                <Badge key={i} variant={c.status === "passed" ? "outline" : "destructive"} className="text-[10px] py-0.5">
                  {c.status === "passed" ? (
                    <ShieldCheck className="mr-1 h-3.5 w-3.5" />
                  ) : (
                    <ShieldAlert className="mr-1 h-3.5 w-3.5" />
                  )}
                  {c.source === "admin" ? "Policy" : "Regulatory"}: {c.status.toUpperCase()}
                </Badge>
              ))}
              {compliance.filter((c: any) => c.source !== "handbook").length === 0 && (
                <span className="text-xs text-muted-foreground italic">No flags.</span>
              )}
            </div>
          </div>

          <div className="flex gap-2 pt-4 border-t">
            <Button variant="outline" size="sm" className="flex-1 gap-2 text-xs" onClick={handleRevert} disabled={loading}>
              <Undo2 className="h-3.5 w-3.5" />
              Revert to Pending
            </Button>
            <Button variant="destructive" size="sm" className="flex-1 gap-2 text-xs" onClick={handleDelete} disabled={loading}>
              <Trash2 className="h-3.5 w-3.5" />
              Delete Permanently
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
