# Feature Specification: Intelligence Hub

**Feature Branch**: `006-intelligence-hub`  
**Created**: 2026-03-20  
**Status**: Ready for Planning  
**Input**: User description: "Create technical specs for Step 6. - UI: Use shadcn/ui Tabs for Chat, Briefings, and Vendor Mode. - Chat Logic: Integrate session memory into the Llama 3.3 prompt chain. - Citations: Return metadata (filename, snippet, page) with every RAG retrieval. - Vendor Mode: Implement a metadata filter in pgvector to isolate vendor-related files. - Briefing: Create a dynamic template engine for executive summaries."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Client Intelligence Chat (Priority: P1)

As a Lawyer, I want to chat with a client's document vault using natural language so that I can quickly retrieve facts and verify information with specific citations.

**Why this priority**: Core value proposition of the Intelligence Hub. Enables the primary interaction for legal research within a case.

**Independent Test**: Can be fully verified by asking a question about a specific document in the vault and receiving a response that includes accurate footnotes linking to the source.

**Acceptance Scenarios**:

1. **Given** a lawyer is in the Intelligence Hub for Client A, **When** they ask "What is the termination period in the service agreement?", **Then** the system returns a response derived from the documents with numbered citations.
2. **Given** an ongoing chat session, **When** the lawyer asks a follow-up question like "And what about the notice period?", **Then** the system uses session memory to understand the context of the previous question and provides a relevant answer.

---

### User Story 2 - Dynamic Executive Briefings (Priority: P2)

As a Lawyer, I want to see a summarized overview of a document that adapts its format based on whether it is a Contract, Evidence, or Pleading, so that I can immediately grasp the most relevant legal points.

**Why this priority**: High utility for rapid document review. Saves time by surfacing key metadata and summaries automatically.

**Independent Test**: Can be verified by switching between different document types and observing the structural change in the "Briefing" tab content.

**Acceptance Scenarios**:

1. **Given** a PDF categorized as a 'Contract', **When** the lawyer views the Briefing tab, **Then** they see a structure focused on 'Parties', 'Term', and 'Key Obligations'.
2. **Given** a document categorized as 'Evidence', **When** the lawyer views the Briefing tab, **Then** they see a structure focused on 'Date of Incident', 'Relevance', and 'Key Quotes'.

---

### User Story 3 - Vendor Intelligence Filter (Priority: P3)

As an Admin or Lawyer, I want to toggle a 'Vendor Mode' to isolate and query only documents related to external vendors across the client's vault, so that I can analyze procurement risks separately from other case data.

**Why this priority**: Essential for large vaults where general queries might be cluttered with non-vendor related evidence.

**Independent Test**: Can be verified by toggling Vendor Mode and performing a search; the results should only include documents tagged as vendor-related in the metadata.

**Acceptance Scenarios**:

1. **Given** Vendor Mode is active, **When** a search query is performed, **Then** the retrieval engine applies a filter to only include documents where `is_vendor = true`.

## Requirements *(mandatory)*

### Clarifications

#### Session 2026-03-20
- Q: Chat Session Persistence Logic → A: Volatile In-Memory only (No database storage for messages; history is lost on refresh).
- Q: "Vendor" Document Identification → A: User Toggle at Upload (Lawyer manually checks a "Vendor Document" box during the upload process).
- Q: Dynamic Briefing Storage/Cache → A: Strictly On-Demand (Briefing is never saved; it is re-generated every time the tab is opened).
- Q: Retrieval Depth & Snippet Volume → A: Balanced (5-7 snippets) passed to the LLM for a single chat query.
- Q: "Empty Retrieval" Handling → A: LLM-Mediated Refusal (If no snippets are found, the LLM explains gracefully that the information is missing from the documents).

### Functional Requirements

- **FR-001**: System MUST provide a tabbed interface (Chat, Briefings, Vendor Mode) using standardized UI components.
- **FR-002**: System MUST integrate full chat session history into the AI prompt chain. Session memory MUST be volatile (cleared upon page refresh or navigation away from the client detail page).
- **FR-003**: AI responses MUST include numbered footnotes that, when clicked, highlight or scroll to the specific source text in the document viewer.
- **FR-004**: Metadata returned with RAG retrieval MUST include: filename, text snippet, and page number. The system MUST retrieve and pass 5-7 relevant snippets to the LLM.
- **FR-005**: System MUST implement a dynamic briefing engine that selects a template based on the document's 'Document Type' property.
- **FR-006**: Briefing templates MUST include specific sections:
  - **Contract**: Parties, Term, Key Obligations.
  - **Evidence**: Date of Incident, Relevance, Key Quotes.
  - **Pleading**: Claims, Parties, Relief.
  - **Correspondence**: Sender, Recipient, Key Demand.
- **FR-007**: Vendor Mode MUST apply a strict filter to pgvector queries to isolate documents based on metadata tags.
- **FR-008**: The system MUST return "I don't know" or a similar statement if the requested information is not found in the documents (Anti-hallucination per Constitution).
- **FR-009**: Footnotes MUST support multiple sources for a single statement. The UI MUST display multiple distinct footnote markers (e.g., [1][2]) for statements derived from multiple segments.

### Key Entities *(include if feature involves data)*

- **BriefingTemplate**: Defines the structure for document summaries. Attributes: `document_type`, `sections` (JSONB configuration of headers/prompts).
- **DocumentMetadata**: Extends existing document records with `is_vendor` (boolean).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can switch between Chat, Briefing, and Vendor Mode tabs in under 100ms.
- **SC-002**: 100% of AI responses in the Intelligence Hub must contain at least one verifiable source citation.
- **SC-003**: Follow-up questions in chat correctly resolve pronouns (e.g., "it", "they") referring to previous answers in 90% of test cases.
- **SC-004**: Briefing generation for a new document completes within 5 seconds of the document being opened for the first time.
- **SC-005**: Vendor Mode filtering reduces the search space by excluding 100% of non-vendor documents during active filtering.

## Assumptions
- Session memory is handled in-memory (client-side or temp server session) and is not persisted in the database across page reloads.
- 'Vendor' documents are identified during the upload process via a manual user toggle/checkbox.
- The Llama 3.3 model on Groq is capable of handling the context window required for session history + RAG snippets.
- Briefings are generated on-the-fly and are not cached on the server or in the database.

## Edge Cases
- **No Documents**: How the Chat UI behaves when a client has zero documents uploaded.
- **Conflicting Metadata**: Briefing engine behavior if a document type is changed while a briefing is being viewed.
- **Empty Retrieval**: The LLM will gracefully explain that the requested information is not present in the documents if the vector search returns no relevant snippets.
