# Feature Specification: AI Contract Review (Step 5)

**Feature Branch**: `005-ai-contract-review`  
**Created**: 2026-03-09  
**Status**: Draft  
**Input**: User description: "Create technical specs for Step 5. - AI Logic: Use Groq Llama-3.3-70b to return structured JSON mapping clauses to risk statuses and rewrites. - UI: Three-pane layout (Document View, Risk List, Action Area) using shadcn/ui. - Integration: Pull Rule context from Step 4 RAG and 'Golden Rules' from the Admin settings. - Features: Side-by-side comparison modal for suggested changes and gap analysis for missing clauses."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Instant Contract Risk Analysis (Priority: P1)

As a Lawyer, I want the system to automatically analyze a contract as soon as I open it in the Review Studio, so I can immediately see where the risks are without manual effort.

**Why this priority**: High. This is the core functionality that provides immediate value and aligns with Constitution Principle XVII.

**Independent Test**: [Describe how this can be verified independently via manual verification (no automated tests allowed per Constitution Principle III) - e.g., "Open a document in Review Studio and verify that a full-document scan triggers automatically without user interaction."]

**Acceptance Scenarios**:

1. **Given** a lawyer has selected a document for review, **When** they enter the Review Studio, **Then** the system MUST initiate a full-document scan immediately.
2. **Given** a scan is in progress, **When** it completes, **Then** the results MUST be displayed using a Traffic Light System (Green/Yellow/Red) for each identified clause.

---

### User Story 2 - Side-by-Side Redlining & Immediate State Update (Priority: P2)

As a Lawyer, I want to compare suggested AI changes with the original text side-by-side and apply them with one click within a rich text editor, so I can efficiently revise the contract manually or via AI.

**Why this priority**: Medium-High. Essential for the "Review" workflow and productivity.

**Independent Test**: [Describe how this can be verified independently via manual verification]

**Acceptance Scenarios**:

1. **Given** a clause has a suggested rewrite, **When** the lawyer clicks 'View Suggestion', **Then** a side-by-side modal MUST show the original and suggested text.
2. **Given** a lawyer is viewing a suggestion, **When** they click 'Accept & Replace', **Then** the document state MUST update immediately in the rich text editor (TipTap).

---

### User Story 3 - Gap Analysis for Missing Clauses (Priority: P3)

As a Lawyer, I want the AI to tell me what is missing from a contract based on the Legal Playbook and Golden Rules, so I don't overlook mandatory requirements.

**Why this priority**: Medium. Crucial for compliance and risk management.

**Independent Test**: [Describe how this can be verified independently via manual verification]

**Acceptance Scenarios**:

1. **Given** a Legal Playbook defines a 'Mandatory' clause (e.g., Termination for Convenience), **When** that clause is missing from the scanned document, **Then** the system MUST flag a 'Red' risk indicating the missing clause.
2. **Given** a Golden Rule specifies a mandatory requirement, **When** the AI performs the gap analysis, **Then** it MUST prioritize the Golden Rule over the Playbook if a conflict arises.

---

### Edge Cases

- What happens when the AI returns malformed JSON or fails to map a clause?
- How does the system handle documents that are too long for a single LLM context window?
- What happens if the 'Golden Rules' are updated while a review is in progress?
- **Concurrent Edits**: Handled via UI warning; however, the last "Save" or "Accept" action will persist.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST perform a full-document scan immediately upon entering the Review Studio (Constitution Principle XVII).
- **FR-002**: AI analysis MUST produce structured results mapping clauses to risk statuses (Green, Yellow, Red) and suggested rewrites (Constitution Principle XVIII).
- **FR-003**: System MUST prioritize Admin-defined 'Golden Rules' over the 'Legal Playbook' during reasoning (Constitution Principle XIX).
- **FR-004**: System MUST perform a 'Gap Analysis' to identify and flag missing mandatory clauses as 'Red' risks (Constitution Principle XX).
- **FR-005**: UI MUST implement a three-pane layout: Document View (Left), Risk List (Middle), Action Area (Right).
- **FR-006**: System MUST provide a side-by-side comparison modal for suggested redlines with an 'Accept & Replace' action. The Review Studio MUST include a rich text editor (TipTap) that allows for both manual edits and immediate updates of AI suggestions (Constitution Principle XXI).
- **FR-007**: AI logic MUST pull context from the client-specific RAG infrastructure (Step 4) to ensure analysis is tailored to the client's historical data and playbook.
- **FR-008**: System MUST maintain a versioned history of all risk analyses, timestamping and preserving each scan result for auditing and review.
- **FR-009**: System MUST allow Lawyers to manually override AI-generated risk statuses (e.g., Red to Green), provided they input a mandatory justification rationale for the override.
- **FR-010**: System MUST show a warning alert if more than one user enters the Review Studio for the same document simultaneously to prevent accidental overwrites (Last Writer Wins).
- **FR-011**: System MUST provide a "Mark as Reviewed" action that transitions the document status to "Reviewed" and logs the final version of the AI/manual analysis.

## Clarifications
### Session 2026-03-09
- Q: Document Editing Capability → A: Rich Text Editor (e.g., TipTap or Quill) for manual and AI-assisted edits.
- Q: Risk Analysis Persistence → A: Versioned History; every scan creates a timestamped version.
- Q: Manual Risk Status Override → A: Full Override allowed with mandatory justification rationale.
- Q: Concurrent Review Handling → A: Last Writer Wins with Warning alert if others are in Review Studio.
- Q: Review Completion Workflow → A: Status Transition; user marks as "Reviewed" to log final analysis.

### Key Entities *(include if feature involves data)*

- **RiskAnalysis**: Represents the result of an AI scan, linked to a Document. Contains a collection of ClauseAnalyses and a timestamp for versioning.
- **ClauseAnalysis**: An individual clause's assessment including risk status (AI-generated or user-overridden), original text, rationale, justification for override (if applicable), and suggested rewrite.
- **GoldenRule**: Global rules set by Admins that govern all AI analysis logic.
- **DocumentState**: The current version of the document being edited in the Review Studio.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: AI Analysis completes and displays initial results in under 15 seconds for a standard 10-page document.
- **SC-002**: 100% of 'Red' risks (High Risk/Prohibited/Missing) are clearly highlighted with actionable rationale.
- **SC-003**: 100% of 'Accept & Replace' actions reflect the change in the Document View and database state instantly.
- **SC-004**: System accurately identifies 100% of missing mandatory clauses defined in the 'Golden Rules'.

## Assumptions

- AI analysis logic will handle text extraction from PDF/DOCX via the existing RAG pipeline.
- 'Golden Rules' will be provided as plain text or structured input in the Admin settings.
- The system will use the established RBAC (Lawyer access only to their clients' documents).
