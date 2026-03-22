import { getFirmContext } from "@/lib/ai/groq";
import { getGroqCompletion } from "@/lib/ai/groq-client";
import { TriageResult } from "@/lib/ai/types";
import { createClient } from "@/lib/supabase/server";

export async function triageDocument(documentId: string): Promise<TriageResult> {
  const supabase = createClient();
  
  const { data: doc } = await supabase
    .from("documents")
    .select("file_name, current_content")
    .eq("id", documentId)
    .single();

  if (!doc) throw new Error("Document not found");

  const documentText = doc.current_content || `Document: ${doc.file_name}. Contents related to legal agreements...`;

  const context = await getFirmContext();
  
  const systemPrompt = `
    You are an AI legal triage assistant. Your goal is to classify a document as 'standard' or 'complex' 
    based on firm Golden Rules and general complexity.
    
    ### FIRM GOLDEN RULES:
    ${context.goldenRules}
    
    ### INSTRUCTIONS:
    1. Analyze the document text.
    2. Assign a complexity score from 1-10 (10 being most complex).
    3. If score >= 7, classify as 'complex'.
    4. Flag compliance issues based on Golden Rules and general regulatory standards (GDPR, CCPA, etc.).
    5. Return a JSON object matching the following structure:
    {
      "classification": "standard" | "complex",
      "complexity_score": number,
      "reasoning": "brief explanation",
      "compliance_flags": [
        {
          "source": "admin" | "regulatory",
          "status": "passed" | "failed",
          "rule_violated": "string or null",
          "description": "string"
        }
      ]
    }
  `;

  const userPrompt = `Document Text (First 10000 characters): \n\n ${documentText.slice(0, 10000)}`;

  const result = await getGroqCompletion(systemPrompt, userPrompt) as TriageResult;

  // Update document with triage results
  await supabase
    .from("documents")
    .update({
      classification: result.classification,
      complexity_score: result.complexity_score,
      triage_reasoning: result.reasoning,
      triage_metadata: { compliance_flags: result.compliance_flags }
    })
    .eq("id", documentId);

  return result;
}
