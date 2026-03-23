import { createClient } from "@/lib/supabase/server";
import { extractTextFromFile } from "@/lib/playbook/parser";
import { searchSimilarContent } from "@/lib/ai/rag";
import { getGroqCompletion } from "@/lib/ai/groq-client";
import { GAP_ANALYSIS_SYSTEM_PROMPT, getGapAnalysisUserPrompt } from "./evolution-prompts";

export async function runGapAnalysis(standardId: string, userId: string) {
  const supabase = createClient();

  // 1. Fetch the Compliance Standard
  const { data: standard, error: sError } = await supabase
    .from("compliance_standards")
    .select("*")
    .eq("id", standardId)
    .single();

  if (sError || !standard) throw new Error("Compliance Standard not found");

  // 2. Download and Extract Text
  const { data: fileData, error: fError } = await supabase.storage
    .from("compliance-standards")
    .download(standard.storage_path);

  if (fError || !fileData) throw new Error("Failed to download standard file");

  const buffer = Buffer.from(await fileData.arrayBuffer());
  const standardText = await extractTextFromFile(buffer, standard.name);
  console.log(`[GapAnalysis] Extracted text length: ${standardText.length}`);

  // 3. Chunk the Standard Text (More robust chunking)
  // Split by double newline first, but if that results in too few chunks, split by single newline and combine
  let chunks = standardText.split("\n\n").filter(c => c.trim().length > 100);
  
  if (chunks.length < 2) {
    // Try single newline if double newline didn't work well
    const lines = standardText.split("\n").filter(l => l.trim().length > 0);
    chunks = [];
    let currentChunk = "";
    for (const line of lines) {
      currentChunk += line + "\n";
      if (currentChunk.length > 1000) {
        chunks.push(currentChunk);
        currentChunk = "";
      }
    }
    if (currentChunk.length > 100) chunks.push(currentChunk);
  }

  console.log(`[GapAnalysis] Number of chunks to process: ${Math.min(chunks.length, 20)} (Total: ${chunks.length})`);

  const allSuggestions = [];

  for (const [index, chunk] of chunks.slice(0, 20).entries()) { // Increased limit
    console.log(`[GapAnalysis] Processing chunk ${index + 1}/${Math.min(chunks.length, 20)}...`);
    // 4. Retrieve Relevant Internal Rules (Stage 1)
    const relevantRules = await searchSimilarContent(chunk, 5);
    console.log(`[GapAnalysis] Found ${relevantRules.length} relevant internal rules.`);
    
    const contextStr = relevantRules.length > 0 
      ? relevantRules
          .map(r => `[${r.metadata?.type || 'rule'}] ID: ${r.metadata?.id || 'unknown'} - ${r.content}`)
          .join("\n\n")
      : "No specific internal rules found for this topic. Suggest new golden rules if the standard requires it.";

    // 5. Compare and Generate Suggestions (Stage 2)
    const systemPrompt = GAP_ANALYSIS_SYSTEM_PROMPT;
    const userPrompt = getGapAnalysisUserPrompt(chunk, contextStr);

    try {
      const response = await getGroqCompletion(systemPrompt, userPrompt, true);
      console.log(`[GapAnalysis] AI returned ${response.suggestions?.length || 0} suggestions for chunk ${index + 1}.`);
      
      if (response.suggestions && Array.isArray(response.suggestions)) {
        // Validate target_id is a UUID or 'new'
        const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        const validSuggestions = response.suggestions.filter(s => 
          s.target_id === 'new' || uuidRegex.test(s.target_id)
        );
        allSuggestions.push(...validSuggestions);
      }
    } catch (err) {
      console.error(`[GapAnalysis] Chunk ${index + 1} Error:`, err);
    }
  }

  console.log(`[GapAnalysis] Total valid suggestions generated: ${allSuggestions.length}`);

  // 6. Save Suggestions to DB
  if (allSuggestions.length > 0) {
    const suggestionsToInsert = allSuggestions.map(s => ({
      standard_id: standardId,
      target_type: s.target_type === 'golden_rule' ? 'golden_rule' : 'playbook',
      target_id: (s.target_id === 'new' || !s.target_id) 
        ? '00000000-0000-0000-0000-000000000000' 
        : s.target_id,
      current_text: s.current_text || null,
      suggested_text: s.suggested_text,
      rationale: s.rationale,
      status: 'pending',
    }));

    const { error: insertError } = await supabase
      .from("policy_suggestions")
      .insert(suggestionsToInsert);

    if (insertError) {
        console.error("Failed to save policy suggestions:", insertError);
    }
  }

  return { 
    suggestionCount: allSuggestions.length 
  };
}
