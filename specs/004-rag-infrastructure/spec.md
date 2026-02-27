# Feature Specification: RAG Infrastructure

**Feature Branch**: `004-rag-infrastructure`  
**Created**: 2026-02-26  
**Status**: Draft  
**Input**: User description: "Create technical specs for Step 4. - Model: mixedbread-ai/mxbai-embed-large-v1 (1024 dimensions). - Database: Enable pgvector extension in Supabase. Create 'embeddings' table with columns: id, document_id, client_id (nullable for Playbook), metadata (jsonb), and embedding (vector(1024)). - Pipeline: Implement a Next.js Edge Function or Server Action that uses 'langchain' or 'ai' SDK to chunk text (RecursiveCharacterTextSplitter) and fetch embeddings from Mixedbread API. - Namespacing: Implement a retrieval function that filters by (client_id = X OR client_id IS NULL) to combine facts and rules."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Semantic Document Search (Priority: P1)

As a lawyer, I want to search through a client's documents using natural language so that I can find relevant legal facts without knowing exact keywords.

**Why this priority**: This is the core value proposition of the RAG system—moving beyond keyword search to semantic understanding.

**Independent Test**: Can be tested by uploading a document about "termination clauses," searching for "ending a contract," and receiving the relevant passage from that document.

**Acceptance Scenarios**:

1. **Given** a lawyer has uploaded a document for Client A, **When** they perform a semantic search for Client A, **Then** they should see relevant text passages from that document.
2. **Given** a lawyer is searching for Client A, **When** relevant "Golden Rules" exist in the Playbook, **Then** those rules should be returned alongside Client A's facts.

---

### User Story 2 - Cross-Client Data Isolation (Priority: P2)

As a lawyer, I want to ensure that my searches for Client A never return information from Client B's documents, maintaining strict confidentiality.

**Why this priority**: Data privacy is a non-negotiable legal requirement.

**Independent Test**: Perform a search while focused on Client A and verify that 0 results from Client B are returned, even if Client B has documents highly relevant to the query string.

**Acceptance Scenarios**:

1. **Given** Document 1 belongs to Client A and Document 2 belongs to Client B, **When** a search is performed for Client A, **Then** Document 2 must never appear in the results.

---

### User Story 3 - Automated Playbook Integration (Priority: P3)

As an admin, I want my uploaded Playbook "Golden Rules" to be automatically vectorized so they are available as context for all lawyers' queries.

**Why this priority**: Ensures consistent firm-wide guidance is applied to every case.

**Independent Test**: Upload a new Playbook rule and verify that a subsequent search by any lawyer includes that rule in the context.

**Acceptance Scenarios**:

1. **Given** a new rule is added to the Playbook (client_id is NULL), **When** any lawyer searches their client's vault, **Then** the new rule should be eligible for retrieval.

## Clarifications

### Session 2026-02-26

- Q: Should the RAG pipeline support scanned PDFs (OCR), or only documents with an existing text layer? → A: Text-only (No OCR).
- Q: Should existing documents from Step 3 be automatically vectorized once the RAG system is deployed? → A: Lazy Vectorization (only vectorize upon re-upload or modification).
- Q: What should be the strategy for semantic retrieval when a user performs a search? → A: Hybrid (Top 5 most relevant results with a 0.7 minimum similarity threshold).
- Q: What should be the chunking configuration for the text splitter? → A: Moderate (500 character chunks with 50 character overlap).
- Q: Should the system provide visual feedback to the user while a document is being vectorized? → A: Status Badge (Async) (updates from "Vectorizing" to "Ready").

## Edge Cases

- **Large Documents**: How does the system handle a 100-page PDF? (Chunks must be correctly sequenced and linked to the source).
- **Scanned PDFs**: Documents without a text layer are out of scope for this phase.
- **Existing Documents**: Documents uploaded prior to Step 4 deployment will not be searchable until they are modified or re-uploaded.
- **Low Relevance Results**: If no matches exceed the 0.7 threshold, the system should inform the user that no relevant passages were found.
- **Empty Documents**: What happens if a document contains no extractable text? (The system should log a warning and skip embedding generation).
- **API Rate Limits**: Handling rate limits from the embedding provider. (Implement retries with exponential backoff).
- **Deletion Sync**: Ensuring embeddings are deleted when a document is removed.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST enable `pgvector` extension in the database to support vector operations.
- **FR-002**: System MUST extract and chunk document text using a recursive character splitting strategy (500 characters per chunk, 50 characters overlap) to preserve context.
- **FR-003**: System MUST generate 1024-dimensional embeddings using the `mixedbread-ai/mxbai-embed-large-v1` model.
- **FR-004**: System MUST store embeddings with associated `document_id` and an optional `client_id` (NULL for global Playbook rules).
- **FR-005**: System MUST implement a retrieval function that filters results based on `client_id` isolation (Specific Client ID OR NULL).
- **FR-006**: System MUST automatically trigger embedding generation upon successful document upload or modification.
- **FR-007**: System MUST provide a search query prefix: "Represent this sentence for searching relevant passages: " for all retrieval queries.
- **FR-008**: System MUST only process documents with an existing text layer (OCR is explicitly excluded).
- **FR-009**: System MUST only return retrieval results with a similarity score >= 0.7, capped at the top 5 most relevant matches.
- **FR-010**: System MUST track the vectorization status of each document (e.g., 'Pending', 'Processing', 'Ready', 'Error').
- **FR-011**: System MUST display the current vectorization status in the document vault UI.

### Key Entities

- **Embedding**: Represents a vector representation of a text chunk. Includes `vector` (1024 dims), `content` (text), `metadata` (JSONB), and references to `document_id` and `client_id`.
- **Document**: Existing entity, now linked to multiple `Embedding` records.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of uploaded text-based documents are successfully vectorized and searchable within 30 seconds of upload.
- **SC-002**: Zero cross-client data leakage (0% of queries return vectors from a different non-null `client_id`).
- **SC-003**: Search results return relevant passages for semantic queries even when exact keyword matches are absent.
- **SC-004**: Retrieval latency for a standard query (top 5 chunks) is under 2 seconds.
- **SC-005**: Query quality is maintained by filtering out noise (all returned results MUST have >= 0.7 similarity score).
