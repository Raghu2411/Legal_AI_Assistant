'use server';

import { Groq } from 'groq-sdk';
import { getTemplateForType, BriefingSection } from '@/lib/ai/briefing-templates';
import { createAdminClient } from '@/lib/supabase/admin';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

export async function generateBriefing(documentId: string, documentType: string) {
  const supabase = createAdminClient();

  try {
    // 1. Fetch document content from embeddings
    const { data: chunks, error: fetchError } = await supabase
      .from('embeddings')
      .select('content')
      .eq('document_id', documentId)
      .order('id', { ascending: true }); // Approximating order by ID if chunk_index is missing

    if (fetchError || !chunks || chunks.length === 0) {
      throw new Error('Document content not found or not yet vectorized.');
    }

    const fullText = chunks.map(c => c.content).join('\n\n');
    const template = getTemplateForType(documentType);

    // 2. Generate briefing for each section
    const briefingResults = await Promise.all(
      template.sections.map(async (section: BriefingSection) => {
        const response = await groq.chat.completions.create({
          model: 'llama-3.3-70b-versatile',
          messages: [
            {
              role: 'system',
              content: `You are a legal AI assistant. Generate a summary for the "${section.title}" section of a ${documentType} briefing.
              Instruction: ${section.instruction}
              CRITICAL: Ground your response strictly in the provided text.
              CRITICAL: Keep it concise (2-3 sentences max).`
            },
            {
              role: 'user',
              content: `DOCUMENT CONTENT:\n${fullText.substring(0, 30000)}` // Basic truncation for context limits
            }
          ],
        });

        return {
          title: section.title,
          content: response.choices[0].message.content || 'No summary generated.'
        };
      })
    );

    return {
      success: true,
      documentType,
      sections: briefingResults
    };

  } catch (error: any) {
    console.error('Error generating briefing:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

export async function getClientDocuments(clientId: string) {
  const supabase = createAdminClient();
  const { data, error } = await supabase
    .from('documents')
    .select('id, file_name, doc_type')
    .eq('client_id', clientId);

  if (error) {
    console.error('Error fetching documents:', error);
    return [];
  }

  // Map doc_type to type for component compatibility
  return (data || []).map(doc => ({
    ...doc,
    type: doc.doc_type
  }));
}
