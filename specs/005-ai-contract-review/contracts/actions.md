# Server Actions: AI Contract Review (Step 5)

## `scanDocument` (AI Review Trigger)
Triggered upon Review Studio entry (Principle XVII).

- **Input**: `documentId: string`, `clientContextId: string` (from Step 4 RAG)
- **Output**: `Promise<{ success: boolean, riskAnalysisId: string, error?: string }>`
- **Workflow**:
    1. Verify `lawyer` role and document ownership.
    2. Fetch document content from Supabase Storage.
    3. Fetch client-specific RAG rules (Step 4) and Admin "Golden Rules."
    4. Call Groq (Llama-3.3-70b) with structured prompt.
    5. Parse and persist `RiskAnalysis` and `ClauseAnalysis` records.
    6. Return result ID for UI mapping.

## `acceptRewrite` (Accept & Replace)
Applies a suggested rewrite to the document state (Principle XXI).

- **Input**: `documentId: string`, `clauseAnalysisId: string`, `rewrite: string`
- **Output**: `Promise<{ success: boolean, updatedContent: string }>`
- **Workflow**:
    1. Update the `DocumentState` (rich text content) by replacing original text with `rewrite`.
    2. Persist the updated state to Supabase.
    3. Mark the specific `ClauseAnalysis` as "Accepted."

## `overrideRiskStatus` (Manual Override)
Allows a lawyer to manually change a risk status with rationale (Principle XX).

- **Input**: `clauseAnalysisId: string`, `newStatus: 'green' | 'yellow' | 'red'`, `rationale: string`
- **Output**: `Promise<{ success: boolean }>`
- **Workflow**:
    1. Update `user_overridden_status` and `user_override_rationale` in `ClauseAnalysis`.
    2. Audit the override for firm-wide reporting.

## `markAsReviewed` (Complete Review)
Transitions document to the "Reviewed" state (Clarification Q5).

- **Input**: `documentId: string`
- **Output**: `Promise<{ success: boolean, documentId: string }>`
- **Workflow**:
    1. Update document status in `documents` table to `reviewed`.
    2. Log the final `RiskAnalysis` version as the "Reviewed Baseline."
