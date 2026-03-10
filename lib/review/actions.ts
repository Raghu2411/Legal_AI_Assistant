"use server";

import { createClient } from "@/lib/supabase/server";
import { RiskStatus } from "@/lib/review/schemas";
import { revalidatePath } from "next/cache";
import { groq } from "@/lib/ai/groq-client";
import { BASE_REVIEW_PROMPT } from "@/lib/ai/review-prompt";
import { parseAIReview } from "@/lib/ai/parser";
import { processDocument } from "@/lib/ai/vector-service";

/**
 * AI Review Trigger: Scans a document for risks using Groq.
 */
export async function scanDocument(documentId: string, clientContextId?: string) {
  const supabase = createClient();
  
  // 1. Verify user
  const { data: { user } } = await supabase.auth.getUser();
  if (!user) return { success: false, error: "Unauthorized" };

  // 2. Fetch document info
  const { data: doc } = await supabase
    .from('documents')
    .select('file_name, review_status')
    .eq('id', documentId)
    .single();

  if (!doc) return { success: false, error: "Document not found" };

  // 3. Update status to 'scanning'
  await supabase
    .from('documents')
    .update({ review_status: 'scanning' })
    .eq('id', documentId);

  try {
    // 4. Fetch document text from embeddings (joined chunks)
    const { data: chunks, error: fetchError } = await supabase
      .from('embeddings')
      .select('content, metadata')
      .eq('document_id', documentId);

    if (fetchError) throw fetchError;
    if (!chunks || chunks.length === 0) {
      throw new Error("Document content not found in embeddings. Please ensure the document has been processed/vectorized.");
    }

    // Sort chunks by index to reconstruct full text
    const fullText = chunks
      .sort((a, b) => (a.metadata?.chunk_index || 0) - (b.metadata?.chunk_index || 0))
      .map(c => c.content)
      .join("\n\n");

    // 5. Fetch Golden Rules
    const { data: goldenRules } = await supabase
      .from('golden_rules')
      .select('rule_text')
      .eq('is_active', true)
      .order('priority', { ascending: false });

    const goldenRulesText = goldenRules?.map(r => r.rule_text).join("\n") || "No Golden Rules defined.";

    // 6. Fetch Playbook Context (Simplified for MVP)
    // In a full RAG implementation, we would use retrieveContext here.
    const playbookContext = "Review against standard legal best practices and firm-wide playbook guidelines.";

    // 7. Call Groq
    const prompt = BASE_REVIEW_PROMPT
      .replace("{{golden_rules}}", goldenRulesText)
      .replace("{{playbook_context}}", playbookContext)
      .replace("{{contract_text}}", fullText);

    const completion = await groq.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      model: "llama-3.3-70b-versatile",
      response_format: { type: "json_object" },
      temperature: 0.1, // Low temperature for consistency
    });

    const rawResult = completion.choices[0]?.message?.content || "";
    const analysisResult = parseAIReview(rawResult);

    // 8. Persist results
    // Create RiskAnalysis session
    const { data: riskAnalysis, error: raError } = await supabase
      .from('risk_analyses')
      .insert({
        document_id: documentId,
        status: 'completed',
        raw_json: analysisResult
      })
      .select()
      .single();

    if (raError) throw raError;

    // Create ClauseAnalysis items
    const clauseRecords = analysisResult.analyses.map(a => ({
      risk_analysis_id: riskAnalysis.id,
      original_text: a.original_text,
      risk_status: a.risk_status,
      rationale: a.rationale,
      suggested_rewrite: a.suggested_rewrite,
      is_gap: a.is_gap
    }));

    const { data: insertedClauses, error: caError } = await supabase
      .from('clause_analyses')
      .insert(clauseRecords)
      .select();

    if (caError) throw caError;

    // 9. Update document status to 'analyzed'
    await supabase
      .from('documents')
      .update({ review_status: 'analyzed' })
      .eq('id', documentId);

    revalidatePath(`/review/${documentId}`);
    
    return { 
      success: true, 
      analysis: {
        ...riskAnalysis,
        clause_analyses: insertedClauses
      }
    };

  } catch (error: any) {
    console.error("Scan Error:", error);
    // Revert status on failure
    await supabase
      .from('documents')
      .update({ review_status: 'uploaded' }) 
      .eq('id', documentId);
    return { success: false, error: error.message || "An error occurred during scanning." };
  }
}

/**
 * Accept & Replace: Applies a suggested rewrite to the document state.
 */
export async function acceptRewrite(documentId: string, clauseAnalysisId: string, rewrite: string) {
  const supabase = createClient();

  // In a real TipTap implementation, the client would handle the string replacement
  // and then call saveDocumentContent.
  // Here we mark the clause as accepted.
  const { error } = await supabase
    .from('clause_analyses')
    .update({
      user_overridden_status: 'green',
      user_override_rationale: 'Accepted AI rewrite.'
    })
    .eq('id', clauseAnalysisId);

  if (error) return { success: false, error: error.message };

  return { success: true, updatedContent: rewrite };
}

/**
 * Persists the current TipTap state to the documents table.
 */
export async function saveDocumentContent(documentId: string, content: string) {
  const supabase = createClient();

  const { error } = await supabase
    .from('documents')
    .update({ current_content: content })
    .eq('id', documentId);

  if (error) return { success: false, error: error.message };

  revalidatePath(`/review/${documentId}`);
  return { success: true };
}

/**
 * Manual Override: Allows a lawyer to manually change a risk status with rationale.
 */
export async function overrideRiskStatus(clauseAnalysisId: string, newStatus: RiskStatus, rationale: string) {
  const supabase = createClient();

  const { error } = await supabase
    .from('clause_analyses')
    .update({
      user_overridden_status: newStatus,
      user_override_rationale: rationale
    })
    .eq('id', clauseAnalysisId);

  if (error) return { success: false, error: error.message };

  return { success: true };
}

/**
 * Complete Review: Transitions document to the "Reviewed" state.
 */
export async function markAsReviewed(documentId: string) {
  const supabase = createClient();

  const { error } = await supabase
    .from('documents')
    .update({ review_status: 'reviewed' })
    .eq('id', documentId);

  if (error) return { success: false, error: error.message };

  revalidatePath(`/review/${documentId}`);
  return { success: true, documentId };
}

/**
 * Re-vectorizes the document using its updated current_content.
 */
export async function revectorizeDocument(documentId: string) {
  const supabase = createClient();

  // 1. Fetch current content and client ID
  const { data: doc, error: fetchError } = await supabase
    .from('documents')
    .select('current_content, client_id')
    .eq('id', documentId)
    .single();

  if (fetchError || !doc) {
    return { success: false, error: "Document not found or has no refined content." };
  }

  if (!doc.current_content) {
    return { success: false, error: "No refined content found. Please save changes in Review Studio first." };
  }

  try {
    // 2. Trigger vectorization using current_content (HTML stripped)
    const cleanText = doc.current_content.replace(/<[^>]*>/g, ' ');
    await processDocument(documentId, doc.client_id, cleanText);
    
    revalidatePath('/vault');
    return { success: true };
  } catch (error: any) {
    console.error("Re-vectorization failed:", error);
    return { success: false, error: error.message || "Failed to re-vectorize document." };
  }
}
