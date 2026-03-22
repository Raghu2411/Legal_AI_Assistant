"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ShieldAlert, ShieldCheck } from "lucide-react";

export function ComplianceSidebar({ metadata }: { metadata: any }) {
  // Filter out Firm Handbook Guidance (source: 'handbook')
  const compliance = (metadata?.compliance || []).filter(
    (c: any) => c.source !== "handbook"
  );

  const getIcon = (source: string, status: string) => {
    return status === "passed" ? (
      <ShieldCheck className="h-4 w-4 text-green-500" />
    ) : (
      <ShieldAlert className="h-4 w-4 text-destructive" />
    );
  };

  const getSourceLabel = (source: string) => {
    if (source === "admin") return "Policy (Golden Rule)";
    return "Regulatory Standard";
  };

  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle className="text-sm font-semibold">Compliance Verification</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {compliance.length === 0 ? (
          <p className="text-xs text-muted-foreground italic">No compliance flags for this item.</p>
        ) : (
          compliance.map((c: any, i: number) => (
            <div key={i} className="space-y-1">
              <div className="flex items-center gap-2">
                {getIcon(c.source, c.status)}
                <span className={`text-xs font-bold uppercase ${c.status === 'deviated' ? 'text-blue-600' : ''}`}>
                  {getSourceLabel(c.source)}: {c.status.toUpperCase()}
                </span>
              </div>
              <p className="text-[10px] text-muted-foreground pl-6">
                {c.reason || c.description || "Compliance check completed."}
              </p>
              {c.rule && (
                <p className="text-[10px] font-mono bg-muted p-1 rounded ml-6">
                  Rule: {c.rule}
                </p>
              )}
            </div>
          ))
        )}
      </CardContent>
    </Card>
  );
}
