"use client";

import React, { useEffect, useState, useCallback, useRef, useMemo } from "react";
import { ThreePaneLayout } from "./layout/three-pane-layout";
import { RiskListPane } from "./layout/risk-list-pane";
import { DocumentPane } from "./layout/document-pane";
import { ActionPane } from "./layout/action-pane";
import { RiskItem } from "./risk-item";
import { TiptapEditor, TiptapEditorRef } from "./editor/tiptap-editor";
import { RedlineModal } from "./redline-modal";
import { 
  scanDocument, 
  acceptRewrite, 
  overrideRiskStatus, 
  saveDocumentContent,
  markAsReviewed 
} from "@/lib/review/actions";
import { ClauseAnalysis, RiskStatus } from "@/lib/review/schemas";
import { Button } from "@/components/ui/button"
import { Loader2, ShieldCheck, PlayCircle, Info, Save, CheckCheck, FileDown, History, Clock } from "lucide-react";
import { createClient } from "@/lib/supabase/client";

interface ReviewStudioProps {
  document: any; 
  initialAnalysis: any;
  documentText: string;
  scanHistory?: any[];
}

export function ReviewStudio({ document: doc, initialAnalysis, documentText, scanHistory = [] }: ReviewStudioProps) {
  const [analysis, setAnalysis] = useState(initialAnalysis);
  const [currentContent, setCurrentContent] = useState(doc.current_content || documentText);
  const [isScanning, setIsScanning] = useState(doc.review_status === "scanning");
  const [selectedRisk, setSelectedRisk] = useState<any | null>(null);
  const [isRedlineOpen, setIsRedlineOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filtering state
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  
  const scanStarted = useRef(false);
  const editorRef = useRef<TiptapEditorRef>(null);
  const supabase = createClient();
  const lastSavedContent = useRef(doc.current_content || documentText);

  // Automatic Scan Trigger (Principle XVII)
  const handleStartScan = useCallback(async () => {
    setIsScanning(true);
    setError(null);
    try {
      const result = await scanDocument(doc.id);
      if (result.success && result.analysis) {
        setAnalysis(result.analysis);
      } else {
        setError(result.error || "Failed to analyze document.");
        scanStarted.current = false;
      }
    } catch (err: any) {
      setError(err.message || "An error occurred.");
      scanStarted.current = false;
    } finally {
      setIsScanning(false);
    }
  }, [doc.id]);

  useEffect(() => {
    if (!analysis && doc.review_status === "uploaded" && !isScanning && !scanStarted.current) {
      scanStarted.current = true;
      handleStartScan();
    }
  }, [analysis, doc.review_status, isScanning, handleStartScan]);

  const handleAcceptRewrite = async () => {
    if (!selectedRisk || !editorRef.current) return;

    try {
      const result = await acceptRewrite(doc.id, selectedRisk.id, selectedRisk.suggested_rewrite);
      if (result.success) {
        if (selectedRisk.is_gap) {
          editorRef.current.appendContent(`<p><strong>${selectedRisk.suggested_rewrite}</strong></p>`);
        } else {
          editorRef.current.replaceText(selectedRisk.original_text, selectedRisk.suggested_rewrite);
        }

        setIsRedlineOpen(false);
        const updatedClauses = (analysis.clause_analyses || []).map((ca: any) => 
          ca.id === selectedRisk.id ? { ...ca, risk_status: 'green' as RiskStatus } : ca
        );
        setAnalysis({ ...analysis, clause_analyses: updatedClauses });
        setSelectedRisk(null);
        
        // Finalize state sync after editor updates - using local state instead of DOM
        // The auto-save effect will handle the actual DB persistence
      }
    } catch (err) {
      alert("Failed to apply suggestion.");
    }
  };

  const handleOverrideStatus = async (status: RiskStatus, rationale: string) => {
    if (!selectedRisk) return;
    if (!rationale) {
      alert("Rationale is required for overrides.");
      return;
    }

    try {
      const result = await overrideRiskStatus(selectedRisk.id, status, rationale);
      if (result.success) {
        const updatedClauses = (analysis.clause_analyses || []).map((ca: any) => 
          ca.id === selectedRisk.id ? { ...ca, risk_status: status } : ca
        );
        setAnalysis({ ...analysis, clause_analyses: updatedClauses });
        setSelectedRisk(null);
      }
    } catch (err) {
      alert("Failed to override status.");
    }
  };

  const handleVersionSwitch = async (analysisId: string) => {
    if (analysisId === "current") return;
    
    setIsScanning(true);
    try {
      const { data, error } = await supabase
        .from('risk_analyses')
        .select('*, clause_analyses(*)')
        .eq('id', analysisId)
        .single();
        
      if (error) throw error;
      setAnalysis(data);
    } catch (err) {
      alert("Failed to load historical scan.");
    } finally {
      setIsScanning(false);
    }
  };

  const filteredRisks = useMemo(() => {
    if (!analysis?.clause_analyses) return [];
    return analysis.clause_analyses.filter((risk: any) => {
      const matchesSearch = risk.original_text.toLowerCase().includes(searchTerm.toLowerCase()) || 
                           risk.rationale.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === "all" || risk.risk_status === statusFilter;
      return matchesSearch && matchesStatus;
    });
  }, [analysis, searchTerm, statusFilter]);

  const handleExport = (type: 'pdf' | 'docx') => {
    if (type === 'pdf') {
      const printWindow = window.open('', '_blank');
      if (printWindow) {
        printWindow.document.write(`
          <html>
            <head><title>${doc.file_name}</title></head>
            <body style="font-family: serif; padding: 40px; line-height: 1.6;">
              <h1>${doc.file_name}</h1>
              <div>${currentContent}</div>
            </body>
          </html>
        `);
        printWindow.document.close();
        printWindow.print();
      }
    } else {
      const blob = new Blob([currentContent.replace(/<[^>]*>/g, ' ')], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download = `${doc.file_name}.txt`;
      window.document.body.appendChild(a);
      a.click();
      window.document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const leftPane = (
    <div className="flex flex-col h-full">
      <div className="p-4 border-b border-border bg-background">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">Scan History</h2>
          <History className="h-4 w-4 text-muted-foreground" />
        </div>
        <select 
          className="w-full bg-muted/50 border-none text-xs p-2 rounded-md focus:ring-1 focus:ring-primary outline-none"
          value={analysis?.id || "current"}
          onChange={(e) => handleVersionSwitch(e.target.value)}
        >
          <option value="current">Current Analysis</option>
          {scanHistory.map((scan) => (
            <option key={scan.id} value={scan.id}>
              {new Date(scan.created_at).toLocaleString()} ({scan.status})
            </option>
          ))}
        </select>
      </div>
      
      <div className="flex-1 overflow-hidden">
        <RiskListPane 
          searchTerm={searchTerm}
          onSearchChange={setSearchTerm}
          statusFilter={statusFilter}
          onStatusFilterChange={setStatusFilter}
        >
          {isScanning && (
            <div className="flex flex-col items-center justify-center p-8 space-y-4 text-center">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
              <p className="text-sm font-medium">Processing...</p>
            </div>
          )}
          
          {!isScanning && filteredRisks.length === 0 && (
            <div className="p-8 text-center space-y-4 opacity-50">
              <Info className="h-8 w-8 mx-auto" />
              <p className="text-sm">No matching risks found.</p>
            </div>
          )}

          {!isScanning && filteredRisks.map((ca: any) => (
            <RiskItem 
              key={ca.id}
              status={ca.risk_status}
              originalText={ca.original_text}
              rationale={ca.rationale}
              suggestedRewrite={ca.suggested_rewrite}
              isGap={ca.is_gap}
              isSelected={selectedRisk?.id === ca.id}
              onClick={() => setSelectedRisk(ca)}
              onViewSuggestion={() => {
                setSelectedRisk(ca);
                setIsRedlineOpen(true);
              }}
            />
          ))}
        </RiskListPane>
      </div>
    </div>
  );

  const centerPane = (
    <div className="flex flex-col h-full">
      <div className="px-4 py-2 border-b border-border bg-background flex items-center justify-between">
        <div className="flex items-center gap-2 overflow-hidden">
          <h2 className="text-sm font-semibold truncate">{doc.file_name}</h2>
          <Badge variant="outline" className="text-[10px] uppercase">{doc.review_status}</Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="h-8" onClick={() => handleExport('pdf')}>
            <FileDown className="h-3.5 w-3.5 mr-1.5" />
            PDF
          </Button>
          <Button variant="outline" size="sm" className="h-8" onClick={() => handleExport('docx')}>
            <FileDown className="h-3.5 w-3.5 mr-1.5" />
            DOCX
          </Button>
          <div className="w-px h-4 bg-border mx-1" />
          <div className="flex items-center gap-2 text-xs text-muted-foreground mr-2">
            {isSaving ? (
              <span className="flex items-center gap-1"><Loader2 className="h-3 w-3 animate-spin" /> Saving...</span>
            ) : (
              <span className="flex items-center gap-1 text-green-600"><CheckCheck className="h-3 w-3" /> Saved</span>
            )}
          </div>
          <Button variant="ghost" size="sm" className="h-8" onClick={() => {
            handleSaveContent(currentContent);
            alert("Changes saved successfully.");
          }}>
            <Save className="h-3.5 w-3.5 mr-1.5" />
            Save Now
          </Button>
        </div>
      </div>
      <div className="flex-1 overflow-hidden">
        <DocumentPane title="">
          <TiptapEditor 
            ref={editorRef}
            content={currentContent} 
            onChange={(val) => {
              setCurrentContent(val);
            }}
          />
        </DocumentPane>
      </div>
    </div>
  );

  const rightPane = (
    <ActionPane 
      selectedRisk={selectedRisk}
      onAcceptSuggestion={() => setIsRedlineOpen(true)}
      onOverrideStatus={handleOverrideStatus}
    />
  );

  return (
    <>
      <ThreePaneLayout 
        leftPane={leftPane}
        centerPane={centerPane}
        rightPane={rightPane}
      />
      
      {selectedRisk && (
        <RedlineModal 
          isOpen={isRedlineOpen}
          onClose={() => setIsRedlineOpen(false)}
          originalText={selectedRisk.original_text}
          suggestedText={selectedRisk.suggested_rewrite || ""}
          onAccept={handleAcceptRewrite}
        />
      )}

      <div className="fixed bottom-0 left-0 right-0 h-12 bg-background border-t border-border flex items-center justify-end px-6 z-10">
        <Button size="sm" className="bg-green-600 hover:bg-green-700" onClick={() => markAsReviewed(doc.id).then(() => alert("Marked as reviewed"))}>
          <CheckCheck className="h-4 w-4 mr-2" />
          Mark as Reviewed
        </Button>
      </div>
    </>
  );
}

function Badge({ children, className, variant = 'default' }: { children: React.ReactNode, className?: string, variant?: 'default' | 'outline' }) {
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 ${variant === 'outline' ? 'text-foreground' : 'border-transparent bg-primary text-primary-foreground hover:bg-primary/80'} ${className}`}>
      {children}
    </span>
  );
}
