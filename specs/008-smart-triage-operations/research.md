# Research: Smart Triage & Operations

## Decision: shadcn/ui Calendar for Operations Dashboard
- **Rationale**: The project already uses shadcn/ui. The Calendar component is highly customizable and integrates well with the existing UI patterns.
- **Alternatives considered**: FullCalendar (rejected as too heavy for the current requirements).

## Decision: Groq/Llama 3.3 for Dual-Scope Compliance Flagging
- **Rationale**: Llama 3.3 is the project standard. We will use a multi-shot prompting technique to provide both Admin Golden Rules and regulatory context (e.g., GDPR, CCPA) for dual-layer flagging.
- **Implementation Pattern**: The AI will return a structured JSON object containing a `compliance` array with `source` ('admin' | 'regulatory') and `status` ('passed' | 'failed') for each check.

## Decision: 'Pending' to 'Confirmed' Workflow for Obligations
- **Rationale**: To satisfy Principle XXXII (Obligation Verification), all AI-extracted milestones must enter a 'Pending' state. 
- **State Transition**: A Lawyer confirmation triggers a database update to 'Confirmed' and creates a corresponding entry in the `activity_logs` for auditability (Principle XXXIV).

## Decision: keyword-based Triage using AI
- **Rationale**: Instead of simple regex, Llama 3.3 will be used to interpret "complexity" based on the semantics of the Golden Rules, even if exact keywords aren't present.
- **Thresholds**: We will implement a "Complexity Score" (1-10) where anything >= 7 is automatically flagged as 'Complex'.
