import { createClient } from "@/lib/supabase/client";
import { Groq } from 'groq-sdk';
import { OpenAIStream, StreamingTextResponse, StreamData } from 'ai';
import { getSystemPrompt, INITIAL_QUESTIONS } from './drafting-prompts';

export interface DraftingSession {
  clientId: string;
  docType: string;
  docName: string;
  startTime: string;
}

export async function logDraftingAction(
  userId: string,
  clientId: string,
  actionType: 'DRAFTING_START' | 'DRAFTING_FINALIZE' | 'EMAIL_GENERATED',
  metadata: any
) {
  const supabase = createClient();
  
  const { error } = await supabase
    .from('activity_logs')
    .insert({
      user_id: userId,
      client_id: clientId,
      action_type: actionType,
      metadata: metadata,
    });

  if (error) {
    console.error(`Error logging drafting action ${actionType}:`, error);
  }
}

export async function draftingChatStream(
  messages: any[],
  session: DraftingSession,
  clientName: string
) {
  const groq = new Groq({
    apiKey: process.env.GROQ_API_KEY,
  });

  const systemPrompt = getSystemPrompt(clientName, session.docType, session.docName);

  // 1. Create stream
  const response = await groq.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: systemPrompt },
      ...messages
    ],
    stream: true,
    temperature: 0.3, // Lower temperature for more stable legal drafting
  });

  const stream = OpenAIStream(response as any);

  return new StreamingTextResponse(stream);
}

export function getInitialGreeting(docType: string): string {
  const questions = INITIAL_QUESTIONS[docType] || ["What are the primary details of this agreement?"];
  return `Hello! I'm ready to help you draft the ${docType}. Let's start with the first question:\n\n${questions[0]}`;
}
