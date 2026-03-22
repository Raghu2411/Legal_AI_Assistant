'use client';

import React, { useEffect, forwardRef, useImperativeHandle } from 'react';
import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import Highlight from '@tiptap/extension-highlight';
import { Card } from '@/components/ui/card';
import { Loader2, Lock } from 'lucide-react';

interface EditorPanelProps {
  content: string;
  onUpdate: (content: string) => void;
  isLocked?: boolean;
}

export interface EditorPanelHandle {
  appendContent: (content: string) => void;
}

const EditorPanel = forwardRef<EditorPanelHandle, EditorPanelProps>(({ content, onUpdate, isLocked }, ref) => {
  const editor = useEditor({
    extensions: [
      StarterKit,
      Highlight.configure({
        multicolor: true,
      }),
    ],
    content: content,
    immediatelyRender: false,
    onUpdate: ({ editor }) => {
      onUpdate(editor.getHTML());
    },
    editable: !isLocked,
  });

  useImperativeHandle(ref, () => ({
    appendContent: (newContent: string) => {
      if (editor) {
        editor.commands.insertContent(newContent);
      }
    }
  }));

  // Sync external content changes (Only if significantly different to avoid loops)
  useEffect(() => {
    if (editor && content !== editor.getHTML()) {
      // We only force set content if it's the initial load or a reset
      // Normal updates should flow through onUpdate or appendContent
      if (editor.isEmpty && content) {
        editor.commands.setContent(content, false);
      }
    }
  }, [content, editor]);

  // Handle locking
  useEffect(() => {
    if (editor) {
      editor.setEditable(!isLocked);
    }
  }, [isLocked, editor]);

  if (!editor) {
    return null;
  }

  return (
    <Card className="flex flex-col h-full border-none rounded-none shadow-none relative bg-white dark:bg-slate-950">
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 lg:p-12 prose prose-sm sm:prose-base max-w-none dark:prose-invert">
        <EditorContent editor={editor} className="min-h-full outline-none" />
      </div>
      {isLocked && (
        <div className="absolute inset-0 bg-background/60 backdrop-blur-[1px] z-50 flex items-center justify-center transition-all p-4">
          <div className="bg-background border shadow-2xl p-4 sm:p-6 rounded-xl flex flex-col items-center gap-3 sm:gap-4 animate-in fade-in zoom-in duration-300 max-w-[280px] sm:max-w-sm text-center">
            <div className="relative">
              <div className="h-12 w-12 sm:h-16 sm:w-16 rounded-full border-2 border-primary/20 flex items-center justify-center">
                <Loader2 className="h-6 w-6 sm:h-8 sm:w-8 text-primary animate-spin" />
              </div>
              <div className="absolute -top-1 -right-1 bg-primary text-primary-foreground p-1 rounded-full shadow-lg">
                <Lock className="h-3 w-3 sm:h-4 sm:w-4" />
              </div>
            </div>
            <div>
              <p className="font-bold text-base sm:text-lg">AI Drafting...</p>
              <p className="text-xs sm:text-sm text-muted-foreground">Editor locked for safety while generating clauses.</p>
            </div>
          </div>
        </div>
      )}
    </Card>
  );
});

EditorPanel.displayName = 'EditorPanel';

export default EditorPanel;
