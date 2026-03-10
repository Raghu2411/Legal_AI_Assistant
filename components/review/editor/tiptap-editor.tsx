"use client";

import { useEditor, EditorContent } from '@tiptap/react';
import StarterKit from '@tiptap/starter-kit';
import React, { useEffect, useImperativeHandle, forwardRef } from 'react';

interface TiptapEditorProps {
  content: string;
  onChange?: (newContent: string) => void;
}

export interface TiptapEditorRef {
  replaceText: (oldText: string, newText: string) => void;
  appendContent: (html: string) => void;
}

export const TiptapEditor = forwardRef<TiptapEditorRef, TiptapEditorProps>(
  ({ content, onChange }, ref) => {
    const editor = useEditor({
      extensions: [
        StarterKit,
      ],
      content: content,
      immediatelyRender: false,
      onUpdate: ({ editor }) => {
        onChange?.(editor.getHTML());
      },
      editorProps: {
        attributes: {
          class: 'prose prose-sm sm:prose lg:prose-lg xl:prose-2xl mx-auto focus:outline-none min-h-[500px]',
        },
      },
    });

    useImperativeHandle(ref, () => ({
      replaceText: (oldText: string, newText: string) => {
        if (!editor) return;
        
        // Use TipTap's internal search and replace logic (simplified)
        const { state, view } = editor;
        const { tr } = state;
        let found = false;

        state.doc.descendants((node, pos) => {
          if (node.isText && node.text?.includes(oldText)) {
            const start = pos + node.text.indexOf(oldText);
            const end = start + oldText.length;
            tr.insertText(newText, start, end);
            found = true;
            return false; // Stop searching once replaced
          }
        });

        if (found) {
          view.dispatch(tr);
        } else {
          // Fallback if exact match fails: append as redline
          editor.commands.insertContent(`<p style="color: green;">[REPLACEMENT SUGGESTED]: ${newText}</p>`);
        }
      },
      appendContent: (html: string) => {
        if (!editor) return;
        editor.commands.focus('end');
        editor.commands.insertContent(html);
      }
    }));

    // Sync content if it changes externally
    useEffect(() => {
      if (editor && content !== editor.getHTML()) {
        editor.commands.setContent(content);
      }
    }, [content, editor]);

    if (!editor) {
      return null;
    }

    return (
      <div className="tiptap-container">
        <EditorContent editor={editor} />
      </div>
    );
  }
);

TiptapEditor.displayName = "TiptapEditor";
