'use client';

import React, { useState, useRef } from 'react';
import { ClientSelector } from './client-selector';
import { DraftingSession, logDraftingAction, getInitialGreeting } from '@/lib/ai/drafting-orchestrator';
import EditorPanel, { EditorPanelHandle } from './editor-panel';
import { ChatPanel } from './chat-panel';
import { Button } from '@/components/ui/button';
import { Save, CheckCircle, Loader2, Mail, MessageSquare, FileText } from 'lucide-react';
import { useRouter } from 'next/navigation';
import { EmailModal } from './email-modal';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

interface DraftingDashboardProps {
  initialClients: any[];
  user: any;
}

export default function DraftingDashboard({ initialClients, user }: DraftingDashboardProps) {
  const [session, setSession] = useState<DraftingSession | null>(null);
  const [docContent, setDocContent] = useState<string>('<h1>New Legal Document</h1><p>Starting AI-assisted drafting...</p>');
  const [isAiLocked, setIsAiLocked] = useState(false);
  const [isFinalizing, setIsFinalizing] = useState(false);
  const [finalizedDocId, setFinalizedDocId] = useState<string | null>(null);
  const [isEmailModalOpen, setIsEmailModalOpen] = useState(false);
  const editorRef = useRef<EditorPanelHandle>(null);
  const router = useRouter();

  const handleStartSession = async (clientId: string, docType: string, docName: string) => {
    const newSession: DraftingSession = {
      clientId,
      docType,
      docName,
      startTime: new Date().toISOString(),
    };
    
    setSession(newSession);

    // T010: Log 'DRAFTING_START'
    await logDraftingAction(user.id, clientId, 'DRAFTING_START', {
      document_type: docType,
      document_name: docName,
    });
  };

  const handleClauseUpdate = (clause: string) => {
    // Append the new clause with a highlight if it contains [MISSING_TERM]
    const styledClause = clause.includes('[MISSING_TERM]') 
      ? clause.replace(/\[MISSING_TERM\]/g, '<mark style="background-color: #fef08a; color: #854d0e;">[MISSING_TERM]</mark>')
      : clause;
      
    const clauseHtml = `<div class="clause-wrapper mb-4">${styledClause}</div>`;
    
    if (editorRef.current) {
      editorRef.current.appendContent(clauseHtml);
    } else {
      setDocContent(prev => prev + clauseHtml);
    }
  };

  const handleFinalize = async () => {
    if (!session) return;
    
    setIsFinalizing(true);
    try {
      const response = await fetch('/api/drafting/finalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          htmlContent: docContent,
          session,
          clientName: initialClients.find(c => c.id === session.clientId)?.name
        })
      });

      if (!response.ok) throw new Error('Failed to finalize document');
      
      const data = await response.json();
      setFinalizedDocId(data.documentId);
    } catch (error) {
      console.error("Finalization error:", error);
      alert("Failed to save document. Please try again.");
    } finally {
      setIsFinalizing(false);
    }
  };

  if (!session) {
    return (
      <div className="h-full bg-accent/20">
        <ClientSelector clients={initialClients} onStart={handleStartSession} />
      </div>
    );
  }

  const clientName = initialClients.find(c => c.id === session.clientId)?.name || 'Unknown Client';

  if (finalizedDocId) {
    return (
      <div className="flex flex-col items-center justify-center h-full p-8 text-center space-y-6 animate-in fade-in duration-500">
        <div className="bg-primary/10 p-6 rounded-full">
          <CheckCircle className="h-16 w-16 text-primary" />
        </div>
        <div className="max-w-md">
          <h2 className="text-3xl font-bold">Document Finalized!</h2>
          <p className="text-muted-foreground mt-2">
            &quot;{session.docName}&quot; has been saved to the client vault and is being indexed for Intelligence Hub.
          </p>
        </div>
        <div className="flex flex-wrap gap-4 justify-center">
          <Button onClick={() => router.push(`/clients/${session.clientId}/vault`)}>
            Go to Vault
          </Button>
          <Button variant="secondary" onClick={() => setIsEmailModalOpen(true)} className="gap-2">
            <Mail className="h-4 w-4" />
            Generate Cover Email
          </Button>
          <Button variant="outline" onClick={() => setSession(null)}>
            Start New Draft
          </Button>
        </div>

        <EmailModal 
          isOpen={isEmailModalOpen}
          onClose={() => setIsEmailModalOpen(false)}
          documentContent={docContent}
          clientName={clientName}
          clientId={session.clientId}
          userId={user.id}
          docName={session.docName}
        />
      </div>
    );
  }

  return (
    <div className="flex flex-col md:flex-row h-full overflow-hidden">
      {/* Mobile/Tablet Tabs Header */}
      <div className="md:hidden border-b bg-background flex items-center justify-between px-4 py-3">
        <div className="flex flex-col">
          <h2 className="font-bold text-sm truncate max-w-[200px]">{session.docName}</h2>
          <p className="text-[10px] text-muted-foreground">{session.docType} for {clientName}</p>
        </div>
        <Button size="sm" variant="outline" onClick={() => setSession(null)} className="h-8 text-xs px-3">
          Cancel
        </Button>
      </div>

      <Tabs defaultValue="chat" className="flex-1 flex flex-col md:hidden overflow-hidden">
        <TabsList className="grid w-full grid-cols-2 h-12 rounded-none border-b bg-muted/20">
          <TabsTrigger value="chat" className="gap-2 data-[state=active]:bg-background">
            <MessageSquare className="h-4 w-4" />
            Interview
          </TabsTrigger>
          <TabsTrigger value="editor" className="gap-2 data-[state=active]:bg-background">
            <FileText className="h-4 w-4" />
            Document
          </TabsTrigger>
        </TabsList>
        <TabsContent value="chat" className="flex-1 overflow-hidden m-0 p-0">
          <ChatPanel 
            session={session} 
            clientName={clientName}
            onClauseUpdate={handleClauseUpdate}
            onAiStateChange={setIsAiLocked}
            initialMessage={getInitialGreeting(session.docType)}
          />
        </TabsContent>
        <TabsContent value="editor" className="flex-1 overflow-hidden m-0 p-0 flex flex-col">
          <div className="px-4 py-2 border-b bg-background flex items-center justify-between sticky top-0 z-10 shadow-sm">
            <span className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider">Preview Mode</span>
            <Button 
              size="sm" 
              onClick={handleFinalize} 
              disabled={isFinalizing || isAiLocked}
              className="h-8 text-xs gap-2"
            >
              {isFinalizing ? <Loader2 className="h-3 w-3 animate-spin" /> : <Save className="h-3 w-3" />}
              Save & Finalize
            </Button>
          </div>
          <div className="flex-1 overflow-y-auto bg-muted/5">
            <EditorPanel 
              ref={editorRef}
              content={docContent} 
              onUpdate={setDocContent} 
              isLocked={isAiLocked} 
            />
          </div>
        </TabsContent>
      </Tabs>

      {/* Desktop Layout (Standard side-by-side) */}
      <div className="hidden md:flex w-full h-full overflow-hidden">
        {/* Sidebar/Chat - Left (Responsive width) */}
        <div className="w-[350px] lg:w-[400px] xl:w-[450px] flex flex-col shadow-sm border-r shrink-0">
          <div className="p-4 border-b bg-accent/10 flex items-center justify-between">
            <div>
              <h2 className="font-bold text-sm truncate max-w-[200px]">{session.docName}</h2>
              <p className="text-[10px] text-muted-foreground">{session.docType} for {clientName}</p>
            </div>
            <button 
              onClick={() => setSession(null)}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors p-1"
            >
              Cancel
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <ChatPanel 
              session={session} 
              clientName={clientName}
              onClauseUpdate={handleClauseUpdate}
              onAiStateChange={setIsAiLocked}
              initialMessage={getInitialGreeting(session.docType)}
            />
          </div>
        </div>

        {/* Editor - Right (60%) */}
        <div className="flex-1 bg-muted/30 overflow-hidden flex flex-col relative">
          <div className="p-4 border-b bg-background flex items-center justify-between shadow-sm z-10">
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold text-muted-foreground tracking-widest uppercase">Editor Sovereignty Mode</span>
            </div>
            <div className="flex items-center gap-2">
              <Button 
                size="sm" 
                onClick={handleFinalize} 
                disabled={isFinalizing || isAiLocked}
                className="gap-2"
              >
                <CheckCircle className="h-4 w-4" />
                Finalize & Upload
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-hidden shadow-inner bg-background m-2 lg:m-4 border rounded-lg overflow-y-auto">
             <EditorPanel 
              ref={editorRef}
              content={docContent} 
              onUpdate={setDocContent} 
              isLocked={isAiLocked} 
            />
          </div>
        </div>
      </div>
    </div>
  );
}
