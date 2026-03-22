'use client';

import { useChat } from 'ai/react';
import { useState, useRef, useEffect } from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Send, User, Bot, Loader2 } from 'lucide-react';
import { CitationBadge } from './citation-badge';
import { cn } from '@/lib/utils';

interface ChatPanelProps {
  clientId: string;
  isVendorOnly?: boolean;
}

export function ChatPanel({ clientId, isVendorOnly = false }: ChatPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  
  const { messages, input, handleInputChange, handleSubmit, isLoading, data } = useChat({
    api: '/api/chat',
    body: {
      clientId,
      isVendorOnly,
    },
  });

  // Automatically scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Extract citations from stream data
  const latestCitations = (data as any[])?.find(d => d.citations)?.citations || [];

  const renderMessageContent = (content: string) => {
    // Regex to find [1], [2], etc.
    const citationRegex = /\[(\d+)\]/g;
    const parts = content.split(citationRegex);
    
    return parts.map((part, i) => {
      // If i is odd, it's a citation index
      if (i % 2 === 1) {
        const index = parseInt(part);
        const citation = latestCitations.find((c: any) => c.index === index);
        
        if (citation) {
          return (
            <CitationBadge
              key={i}
              index={index}
              snippet={citation.content}
              sourceName={citation.metadata?.source_name || `Source ${index}`}
              onClick={() => {
                // T012 logic: highlight source text
                const event = new CustomEvent('highlight-citation', {
                  detail: { document_id: citation.document_id, snippet: citation.content }
                });
                window.dispatchEvent(event);
              }}
            />
          );
        }
        return `[${part}]`;
      }
      return part;
    });
  };

  return (
    <div className="flex flex-col h-full">
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="flex flex-col gap-4 max-w-3xl mx-auto">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center gap-2">
              <Bot className="h-12 w-12 text-muted-foreground/50" />
              <h3 className="text-lg font-semibold">Intelligence Chat</h3>
              <p className="text-sm text-muted-foreground max-w-xs">
                Ask anything about this client's documents. Responses are grounded in the vault.
              </p>
            </div>
          )}
          
          {messages.map((m) => (
            <div
              key={m.id}
              className={cn(
                "flex gap-3 text-sm",
                m.role === 'user' ? "flex-row-reverse" : ""
              )}
            >
              <div className={cn(
                "flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow",
                m.role === 'user' ? "bg-background" : "bg-primary text-primary-foreground"
              )}>
                {m.role === 'user' ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
              </div>
              <div className={cn(
                "rounded-lg px-4 py-2 max-w-[85%]",
                m.role === 'user' ? "bg-muted" : "bg-background border shadow-sm"
              )}>
                <div className="whitespace-pre-wrap leading-relaxed">
                  {renderMessageContent(m.content)}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && messages[messages.length - 1]?.role === 'user' && (
            <div className="flex gap-3 text-sm">
              <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-md border shadow bg-primary text-primary-foreground">
                <Bot className="h-4 w-4" />
              </div>
              <div className="rounded-lg px-4 py-2 bg-background border shadow-sm flex items-center">
                <Loader2 className="h-4 w-4 animate-spin mr-2" />
                Thinking...
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="p-4 border-t bg-background">
        <form
          onSubmit={handleSubmit}
          className="flex gap-2 max-w-3xl mx-auto items-center"
        >
          <Input
            value={input}
            onChange={handleInputChange}
            placeholder="Ask about these documents..."
            className="flex-1"
            disabled={isLoading}
          />
          <Button type="submit" size="icon" disabled={isLoading || !input.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-[10px] text-center text-muted-foreground mt-2">
          AI-generated responses. Always verify legal findings.
        </p>
      </div>
    </div>
  );
}
