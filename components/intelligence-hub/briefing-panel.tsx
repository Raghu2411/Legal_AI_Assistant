'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Loader2, FileText, RefreshCw, AlertCircle } from 'lucide-react';
import { generateBriefing, getClientDocuments } from '@/app/(lawyer)/intelligence-hub/actions';
import { cn } from '@/lib/utils';

interface BriefingPanelProps {
  clientId: string;
}

export function BriefingPanel({ clientId }: BriefingPanelProps) {
  const [documents, setDocuments] = useState<any[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [briefing, setBriefing] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchDocs() {
      const docs = await getClientDocuments(clientId);
      setDocuments(docs);
      if (docs.length > 0) {
        setSelectedDocId(docs[0].id);
      }
    }
    fetchDocs();
  }, [clientId]);

  const handleGenerate = async () => {
    if (!selectedDocId) return;
    
    setIsGenerating(true);
    setError(null);
    setBriefing(null);
    
    const doc = documents.find(d => d.id === selectedDocId);
    try {
      const result = await generateBriefing(selectedDocId, doc.type || 'Correspondence');
      if (result.success) {
        setBriefing(result);
      } else {
        setError(result.error);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b bg-muted/30 flex items-center justify-between gap-4">
        <div className="flex-1 max-w-sm">
          <select
            className="w-full bg-background border rounded-md p-2 text-sm focus:ring-1 focus:ring-primary outline-none"
            value={selectedDocId}
            onChange={(e) => setSelectedDocId(e.target.value)}
            disabled={isGenerating}
          >
            {documents.length === 0 && <option value="">No documents available</option>}
            {documents.map((doc) => (
              <option key={doc.id} value={doc.id}>
                {doc.file_name} ({doc.type || 'Other'})
              </option>
            ))}
          </select>
        </div>
        <Button 
          onClick={handleGenerate} 
          disabled={isGenerating || !selectedDocId}
          size="sm"
        >
          {isGenerating ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <RefreshCw className="h-4 w-4 mr-2" />
          )}
          Generate Briefing
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 bg-muted/10">
        <div className="max-w-4xl mx-auto space-y-6 pb-8">
          {isGenerating && (
            <div className="flex flex-col items-center justify-center py-20 gap-4">
              <Loader2 className="h-10 w-10 animate-spin text-primary" />
              <div className="text-center">
                <h3 className="font-semibold text-lg">Synthesizing Briefing...</h3>
                <p className="text-sm text-muted-foreground">Analyzing document structure and extracting key sections.</p>
              </div>
            </div>
          )}

          {!isGenerating && !briefing && !error && (
            <div className="flex flex-col items-center justify-center py-20 text-center gap-4 opacity-50">
              <FileText className="h-16 w-16" />
              <div>
                <h3 className="text-lg font-semibold">Ready to Analyze</h3>
                <p className="text-sm max-w-xs">Select a document above to generate an on-demand executive summary based on its type.</p>
              </div>
            </div>
          )}

          {error && (
            <div className="flex items-center gap-3 p-4 rounded-lg bg-destructive/10 text-destructive border border-destructive/20 max-w-lg mx-auto mt-10">
              <AlertCircle className="h-5 w-5 shrink-0" />
              <p className="text-sm font-medium">{error}</p>
            </div>
          )}

          {briefing && (
            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-2 duration-500">
              <div className="flex items-center justify-between">
                <h2 className="text-2xl font-bold tracking-tight text-primary">Executive Summary</h2>
                <Badge variant="outline" className="uppercase font-bold tracking-tighter bg-primary/5">
                  {briefing.documentType}
                </Badge>
              </div>
              
              <div className="grid gap-4">
                {briefing.sections.map((section: any, idx: number) => (
                  <Card key={idx} className="border-none shadow-sm ring-1 ring-border/50">
                    <CardHeader className="py-3 px-4 bg-muted/20">
                      <CardTitle className="text-sm font-bold uppercase tracking-widest text-muted-foreground">
                        {section.title}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="py-3 px-4">
                      <p className="text-sm leading-relaxed whitespace-pre-wrap">
                        {section.content}
                      </p>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
