'use client';

import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useChat } from 'ai/react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { DraftingSession } from '@/lib/ai/drafting-orchestrator';
import { Send, Loader2, Sparkles } from 'lucide-react';

interface ChatPanelProps {
  session: DraftingSession;
  clientName: string;
  onClauseUpdate: (clause: string) => void;
  onAiStateChange: (isLocked: boolean) => void;
  initialMessage?: string;
}

export const ChatPanel: React.FC<ChatPanelProps> = ({
  session,
  clientName,
  onClauseUpdate,
  onAiStateChange,
  initialMessage,
}) => {
  const [processedClauses, setProcessedClauses] = useState<Set<string>>(new Set());
  
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: '/api/drafting/chat',
    body: {
      session,
      clientName,
    },
    initialMessages: initialMessage ? [
      { id: 'initial', role: 'assistant', content: initialMessage }
    ] : [],
    onResponse: () => {
      onAiStateChange(true);
      setProcessedClauses(new Set()); // Reset for new message
    },
    onFinish: () => {
      onAiStateChange(false);
    },
  });

  const scrollRef = useRef<HTMLDivElement>(null);

  const extractNewClauses = useCallback((text: string) => {
    // Regex for fully captured clauses: [[CLAUSE: content ]]
    const clauseRegex = /\[\[CLAUSE:\s*([\s\S]*?)\]\]/g;
    let match;
    const newProcessed = new Set(processedClauses);
    let updated = false;

    while ((match = clauseRegex.exec(text)) !== null) {
      const fullMatch = match[0];
      const clauseContent = match[1].trim();

      // Only process if we haven't seen this specific clause instance in the current message
      // We use the full match as key to ensure uniqueness if multiple identical clauses are sent
      if (!newProcessed.has(fullMatch)) {
        onClauseUpdate(clauseContent);
        newProcessed.add(fullMatch);
        updated = true;
      }
    }

    if (updated) {
      setProcessedClauses(newProcessed);
    }
  }, [processedClauses, onClauseUpdate]);

  // Incremental Clause Extraction
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'assistant') {
        extractNewClauses(lastMessage.content);
      }
    }
  }, [messages, extractNewClauses]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full bg-background border-r overflow-hidden">
      <div className="p-3 bg-primary/5 border-b flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-primary animate-pulse" />
        <span className="text-xs font-semibold text-primary/80 uppercase tracking-wider">Interview Mode</span>
      </div>
      
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-4 pb-4">
          {messages.map((m) => {
            // Filter out clauses from the display for a cleaner experience
            const displayContent = m.content.replace(/\[\[CLAUSE:[\s\S]*?\]\]/g, '').trim();
            
            if (!displayContent && m.role === 'assistant' && isLoading) return null;

            return (
              <div
                key={m.id}
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
              >
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                    m.role === 'user'
                      ? 'bg-primary text-primary-foreground rounded-tr-none shadow-md'
                      : 'bg-muted border border-border/50 shadow-sm rounded-tl-none'
                  }`}
                >
                  <div className="whitespace-pre-wrap leading-relaxed">
                    {displayContent || (isLoading && m.role === 'assistant' ? "..." : "")}
                  </div>
                </div>
              </div>
            );
          })}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-2xl rounded-tl-none px-4 py-2.5 text-sm flex items-center gap-3 border border-border/50 shadow-sm animate-pulse">
                <Loader2 className="h-3 w-3 animate-spin text-primary" />
                <span className="text-muted-foreground italic">Thinking...</span>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>
      
      <div className="p-4 border-t bg-accent/5 backdrop-blur-sm">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Answer the AI's question..."
            disabled={isLoading}
            className="flex-1 bg-background shadow-inner"
          />
          <Button type="submit" size="icon" disabled={isLoading || !input.trim()} className="rounded-full h-10 w-10 shrink-0 shadow-md">
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-[10px] text-muted-foreground mt-3 text-center flex items-center justify-center gap-1">
          <Sparkles className="h-3 w-3" />
          AI-guided drafting is active
        </p>
      </div>
    </div>
  );
};
