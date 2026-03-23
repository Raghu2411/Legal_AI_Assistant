"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Loader2, RotateCcw, ChevronDown, ChevronUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "sonner";

export function HistoryView() {
  const [history, setHistory] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRollingBack, setIsRollingBack] = useState<string | null>(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/evolution/history");
      if (!response.ok) throw new Error("Failed to fetch history");
      const data = await response.json();
      setHistory(data.history || []);
    } catch (err) {
      toast.error("Could not load audit trail.");
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  const handleRollback = async (historyId: string) => {
    setIsRollingBack(historyId);
    try {
      const response = await fetch("/api/evolution/rollback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ historyId }),
      });

      if (!response.ok) throw new Error("Rollback failed");

      toast.success("Successfully rolled back to previous version.");
      fetchHistory();
    } catch (err) {
      toast.error("Failed to perform rollback.");
    } finally {
      setIsRollingBack(null);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
        <span className="ml-2 text-lg">Loading audit trail...</span>
      </div>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Immutable Policy Audit Trail</CardTitle>
        <CardDescription>
          Track every change made to firm playbooks and golden rules.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Timestamp</TableHead>
              <TableHead>Type</TableHead>
              <TableHead>Action</TableHead>
              <TableHead>Changes</TableHead>
              <TableHead>User</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {history.map((item) => (
              <TableRow key={item.id}>
                <TableCell className="text-xs">
                  {new Date(item.timestamp).toLocaleString()}
                </TableCell>
                <TableCell>
                  <Badge variant="outline">{item.entity_type}</Badge>
                </TableCell>
                <TableCell>
                  <Badge variant={item.change_type === 'rollback' ? 'destructive' : 'default'}>
                    {item.change_type}
                  </Badge>
                </TableCell>
                <TableCell>
                  <div className="max-w-xs space-y-1">
                    <p className="text-xs font-semibold text-muted-foreground uppercase">Before:</p>
                    <p className="text-xs truncate bg-red-50 p-1 line-through">
                      {JSON.stringify(item.old_value)}
                    </p>
                    <p className="text-xs font-semibold text-muted-foreground uppercase">After:</p>
                    <p className="text-xs truncate bg-green-50 p-1">
                      {JSON.stringify(item.new_value)}
                    </p>
                  </div>
                </TableCell>
                <TableCell className="text-xs">{item.user_id.slice(0, 8)}</TableCell>
                <TableCell className="text-right">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRollback(item.id)}
                    disabled={isRollingBack === item.id || item.change_type === 'rollback'}
                  >
                    {isRollingBack === item.id ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <RotateCcw className="h-4 w-4" />
                    )}
                    <span className="ml-2">Rollback</span>
                  </Button>
                </TableCell>
              </TableRow>
            ))}
            {history.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="text-center py-10 text-muted-foreground">
                  No version history found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}
