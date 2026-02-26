import { Groq } from "groq-sdk";
import { createClient } from "@/lib/supabase/server";

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

/**
 * Retrieves the latest firm-wide context (Golden Rules + Playbook Text).
 */
export async function getFirmContext() {
  const supabase = createClient();

  const { data: latest } = await supabase
    .from("playbooks")
    .select("golden_rules, file_name")
    .order("version", { ascending: false })
    .limit(1)
    .single();

  return {
    goldenRules: latest?.golden_rules || "No Golden Rules defined.",
    playbookName: latest?.file_name || "No Playbook uploaded.",
    // Note: In Step 2, we assume the extracted text is in golden_rules or handled similarly
    // For a real app, you might store the extracted PDF text in a separate field
  };
}

/**
 * Generates a legal response using Llama 3.3 via Groq, 
 * incorporating firm-wide context and explicit citations.
 */
export async function generateLegalResponse(userQuery: string) {
  const context = await getFirmContext();

  const systemPrompt = `
    You are SAI-Legal, a highly specialized legal assistant. 
    Your responses MUST adhere to the firm's specific guidelines provided below.

    ### FIRM CONTEXT:
    1. **Golden Rules (High Priority):**
    ${context.goldenRules}

    2. **Active Playbook:**
    Source: ${context.playbookName}

    ### INSTRUCTIONS:
    - If the user query relates to firm procedures, reference the Golden Rules or Playbook explicitly.
    - **Citations:** Use format "Per Golden Rules..." or "According to Playbook...".
    - **Conflict Detection:** If a Golden Rule contradicts common legal practice or the Playbook, you MUST highlight this conflict to the user.
    - Maintain a professional, concise, and objective tone.
  `;

  try {
    const chatCompletion = await groq.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userQuery },
      ],
      model: "llama-3.3-70b-versatile",
      temperature: 0.2, // Low temperature for legal consistency
    });

    return {
      content: chatCompletion.choices[0]?.message?.content || "No response generated.",
      sources: [
        { name: "Golden Rules", type: "text" },
        { name: context.playbookName, type: "file" }
      ]
    };
  } catch (error) {
    console.error("Groq AI Error:", error);
    throw new Error("Failed to generate AI response");
  }
}
