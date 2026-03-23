"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Upload, FileText, CheckCircle2, Loader2, RefreshCw } from "lucide-react";
import { toast } from "sonner";

interface AuditPanelProps {
  initialStandards: any[];
  selectedStandardId: string | null;
  onStandardSelect: (id: string) => void;
  playbooks: any[];
}

export function AuditPanel({
  initialStandards,
  selectedStandardId,
  onStandardSelect,
  playbooks,
}: AuditPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [standards, setStandards] = useState(initialStandards);
  const [isRefreshing, setIsRefreshing] = useState<string | null>(null);

  const handleRefreshPlaybook = async (id: string) => {
    setIsRefreshing(id);
    try {
      const response = await fetch("/api/generation/refresh", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ playbookId: id }),
      });
      if (!response.ok) throw new Error("Refresh failed");
      toast.success("Playbook generation and RAG sync started.");
    } catch (err) {
      toast.error("Failed to refresh playbook.");
    } finally {
      setIsRefreshing(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setIsUploading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("name", file.name);

    try {
      const response = await fetch("/api/evolution/audit", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const result = await response.json();
      toast.success(`Analysis complete. Generated ${result.suggestionCount || 0} suggestions.`);
      
      // Update local list
      const newStandard = {
        id: result.standardId,
        name: file.name,
        uploaded_at: new Date().toISOString(),
      };
      setStandards([newStandard, ...standards]);
      onStandardSelect(result.standardId);
      setFile(null);
    } catch (err) {
      toast.error("Failed to upload compliance standard.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
      <Card className="col-span-3">
        <CardHeader>
          <CardTitle>Upload Compliance Standard</CardTitle>
          <CardDescription>
            Upload regulatory documents or external standards to audit against your firm&apos;s rules.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid w-full items-center gap-1.5">
            <Label htmlFor="standard-file">Select PDF or DOCX</Label>
            <Input 
              id="standard-file" 
              type="file" 
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              accept=".pdf,.docx"
            />
          </div>
          <Button 
            className="w-full" 
            disabled={!file || isUploading}
            onClick={handleUpload}
          >
            {isUploading ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Upload className="mr-2 h-4 w-4" />
                Start Gap Analysis
              </>
            )}
          </Button>
        </CardContent>
      </Card>

      <div className="col-span-4 space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Active Playbooks</CardTitle>
            <CardDescription>
              Generate updated DOCX artifacts and sync with RAG.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {playbooks.map((p) => (
              <div key={p.id} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex items-center space-x-3">
                  <FileText className="h-5 w-5 text-blue-600" />
                  <div>
                    <p className="text-sm font-medium">{p.name}</p>
                    <p className="text-xs text-muted-foreground">Version {p.version}</p>
                  </div>
                </div>
                <Button 
                  size="sm" 
                  variant="outline"
                  onClick={() => handleRefreshPlaybook(p.id)}
                  disabled={isRefreshing === p.id}
                >
                  {isRefreshing === p.id ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  <span className="ml-2">Sync RAG</span>
                </Button>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Standards</CardTitle>
            <CardDescription>
              Click on a standard to view AI-generated suggestions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {standards.map((s) => (
                <div
                  key={s.id}
                  className={`flex items-center justify-between p-3 border rounded-lg cursor-pointer transition-colors ${
                    selectedStandardId === s.id ? "bg-primary/5 border-primary" : "hover:bg-muted"
                  }`}
                  onClick={() => onStandardSelect(s.id)}
                >
                  <div className="flex items-center space-x-3">
                    <FileText className="h-5 w-5 text-muted-foreground" />
                    <div>
                      <p className="text-sm font-medium leading-none">{s.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {new Date(s.uploaded_at).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  {selectedStandardId === s.id && (
                    <CheckCircle2 className="h-4 w-4 text-primary" />
                  )}
                </div>
              ))}
              {standards.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  No standards uploaded yet.
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
