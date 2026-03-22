import { getFirmContext } from "@/lib/ai/groq";
import { getGroqCompletion } from "@/lib/ai/groq-client";
import { ExtractedObligation } from "@/lib/ai/types";
import { createClient } from "@/lib/supabase/server";

export async function extractObligations(documentId: string): Promise<ExtractedObligation[]> {
  const supabase = createClient();
  
  const { data: doc } = await supabase
    .from("documents")
    .select("file_name, current_content, client_id")
    .eq("id", documentId)
    .single();

  if (!doc) throw new Error("Document not found");

  const documentText = doc.current_content || `Document: ${doc.file_name}.`;
  const context = await getFirmContext();

  const systemPrompt = `
    You are an AI legal expert specializing in obligation extraction and multi-scope compliance.
    
    ### FIRM CONTEXT:
    1. **GOLDEN RULES (Mandatory - Policy Scope):**
    ${context.goldenRules}

    2. **FIRM HANDBOOK (Guidance - Standard Positions):**
    ${context.handbookText.slice(0, 10000)} (Truncated if necessary)
    
    ### INSTRUCTIONS:
    1. Extract all legal obligations, milestones, and tasks from the document.
    2. For each obligation, perform a TRIPLE-SCOPE compliance flagging:
       - **Policy Layer (Admin)**: Check against Mandatory Golden Rules. Mark as 'failed' if a rule is explicitly violated.
       - **Guidance Layer (Handbook)**: Check against the Firm Handbook. Mark as 'deviated' if the clause differs from the firm's standard position or recommendations.
       - **Regulatory Layer**: Check against general legal standards (GDPR, CCPA, etc.).
    3. Return a JSON object with an "obligations" array matching this structure:
    {
      "obligations": [
        {
          "description": "string",
          "due_date": "ISO-8601 string or null",
          "is_tbd": boolean,
          "metadata": {
            "compliance": [
              { "source": "admin", "status": "passed" | "failed", "rule": "string", "reason": "string" },
              { "source": "handbook", "status": "passed" | "deviated", "reason": "string" },
              { "source": "regulatory", "status": "passed" | "failed", "reason": "string" }
            ]
          }
        }
      ]
    }
  `;

  const userPrompt = `Document Text (First 15000 characters): \n\n ${documentText.slice(0, 15000)}`;

  const result = await getGroqCompletion(systemPrompt, userPrompt);
  const extracted = (result.obligations || []) as ExtractedObligation[];

  if (extracted.length > 0) {
    const obligationsToInsert = extracted.map(ob => ({
      document_id: documentId,
      client_id: doc.client_id,
      description: ob.description,
      due_date: ob.due_date,
      status: 'pending',
      metadata: ob.metadata
    }));

    await supabase.from("obligations").insert(obligationsToInsert);
  }

  return extracted;
}
