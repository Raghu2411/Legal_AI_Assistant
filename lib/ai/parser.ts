import { riskAnalysisResponseSchema, RiskAnalysisResponse } from "@/lib/review/schemas";

/**
 * Parses raw AI string output into structured JSON.
 * Handles common LLM formatting issues like markdown code blocks.
 */
export function parseAIReview(rawOutput: string): RiskAnalysisResponse {
  try {
    // 1. Remove markdown code blocks if present (```json ... ```)
    const jsonMatch = rawOutput.match(/```(?:json)?\s*([\s\S]*?)```/);
    const cleanContent = jsonMatch ? jsonMatch[1].trim() : rawOutput.trim();

    // 2. Parse JSON
    const parsed = JSON.parse(cleanContent);

    // 3. Validate with Zod
    return riskAnalysisResponseSchema.parse(parsed);
  } catch (error) {
    console.error("Failed to parse AI review output:", error);
    console.error("Raw Output was:", rawOutput);
    throw new Error("Invalid AI analysis format. Please try scanning again.");
  }
}
