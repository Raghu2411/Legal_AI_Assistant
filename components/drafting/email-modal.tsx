'use client';

import React, { useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Loader2, Mail, Copy, Check } from 'lucide-react';
import { logDraftingAction } from '@/lib/ai/drafting-orchestrator';

interface EmailModalProps {
  isOpen: boolean;
  onClose: () => void;
  documentContent: string;
  clientName: string;
  clientId: string;
  userId: string;
  docName: string;
}

export const EmailModal: React.FC<EmailModalProps> = ({
  isOpen,
  onClose,
  documentContent,
  clientName,
  clientId,
  userId,
  docName,
}) => {
  const [emailContent, setEmailContent] = useState<string>('');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const generateEmail = async () => {
    setIsGenerating(true);
    try {
      const response = await fetch('/api/drafting/email', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          documentContent,
          clientName,
          docName,
        }),
      });

      if (!response.ok) throw new Error('Failed to generate email');
      
      const data = await response.json();
      setEmailContent(data.emailContent);

      // T023: Log 'EMAIL_GENERATED'
      await logDraftingAction(userId, clientId, 'EMAIL_GENERATED', {
        document_name: docName,
        recipient: clientName,
      });
    } catch (error) {
      console.error("Email generation error:", error);
      alert("Failed to generate email preview.");
    } finally {
      setIsGenerating(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(emailContent);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Mail className="h-5 w-5 text-primary" />
            Generate Cover Email
          </DialogTitle>
          <DialogDescription>
            AI will draft a professional cover email based on the finalized document.
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-4 py-4">
          {!emailContent && !isGenerating ? (
            <div className="flex flex-col items-center justify-center py-12 border-2 border-dashed rounded-lg space-y-4">
              <p className="text-sm text-muted-foreground">Ready to draft the cover email for {clientName}?</p>
              <Button onClick={generateEmail}>
                Generate Email Draft
              </Button>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-xs font-medium text-muted-foreground">Draft Email</span>
                <Button variant="ghost" size="sm" onClick={copyToClipboard} disabled={!emailContent}>
                  {isCopied ? <Check className="h-4 w-4 mr-1 text-green-500" /> : <Copy className="h-4 w-4 mr-1" />}
                  {isCopied ? 'Copied' : 'Copy'}
                </Button>
              </div>
              <Textarea 
                value={emailContent} 
                onChange={(e) => setEmailContent(e.target.value)}
                placeholder={isGenerating ? "Generating..." : ""}
                className="min-h-[300px] text-sm font-sans leading-relaxed"
                disabled={isGenerating}
              />
              {isGenerating && (
                <div className="absolute inset-0 bg-background/50 flex items-center justify-center rounded-lg">
                  <Loader2 className="h-8 w-8 animate-spin text-primary" />
                </div>
              )}
            </div>
          )}
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          {emailContent && (
            <Button onClick={copyToClipboard} className="gap-2">
              <Copy className="h-4 w-4" />
              Copy for Email Client
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};
