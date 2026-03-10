import { z } from "zod";

export const riskStatusSchema = z.enum(["green", "yellow", "red"]);

export const clauseAnalysisSchema = z.object({
  original_text: z.string().describe("The relevant text snippet from the document"),
  risk_status: riskStatusSchema.describe("Risk level assigned to the clause"),
  rationale: z.string().describe("AI explanation for the risk status"),
  suggested_rewrite: z.string().nullable().optional().describe("Proposed replacement text for high-risk clauses"),
  is_gap: z.boolean().default(false).describe("True if this identifies a missing mandatory clause"),
});

export const riskAnalysisResponseSchema = z.object({
  analyses: z.array(clauseAnalysisSchema),
  overall_summary: z.string().describe("A high-level summary of the contract risks"),
});

export type ClauseAnalysis = z.infer<typeof clauseAnalysisSchema>;
export type RiskAnalysisResponse = z.infer<typeof riskAnalysisResponseSchema>;
export type RiskStatus = z.infer<typeof riskStatusSchema>;
