# Tasks: RAG Infrastructure

**Branch**: `004-rag-infrastructure`  
**Status**: 2026-02-26  
**Spec**: [specs/004-rag-infrastructure/spec.md](C:\Users\USER\Desktop\Legal_AI_Assistant\specs\004-rag-infrastructure\spec.md)  
**Plan**: [specs/004-rag-infrastructure/plan.md](C:\Users\USER\Desktop\Legal_AI_Assistant\specs\004-rag-infrastructure\plan.md)

## Implementation Strategy

Build the foundational vector storage and embedding pipeline first. Then implement semantic retrieval as the MVP (User Story 1), followed by strict data isolation (User Story 2) and Playbook integration (User Story 3). Polish with status tracking and UI badges.

## Parallel Execution

- Database migrations (T003, T004) can be prepared alongside utility implementation (T005, T006).
- Search UI (T010) can be drafted while retrieval logic (T009) is being developed.

## Dependencies

- **Phase 2 (Foundational)** must be completed before any User Story implementation.
- **US2 (Data Isolation)** relies on the RLS policies and retrieval filters established in **US1**.

## Phase 1: Setup

Story Goal: Initialize project and install dependencies
Independent Test: Environment variables are loaded and `npm install` completes successfully.

- [x] T001 Add `MIXEDBREAD_API_KEY` to `C:\Users\USER\Desktop\Legal_AI_Assistant\.env.local`
- [x] T002 [P] Install dependencies: `mixedbread-ai`, `langchain`, `pdf-parse`, `mammoth` via `package.json`

## Phase 2: Foundational

Story Goal: Establish vector storage and basic embedding generation pipeline
Independent Test: `embeddings` table exists with HNSW index; `processDocument` generates vectors for a sample string.

- [x] T003 Create migration for `pgvector` and `embeddings` table in `C:\Users\USER\Desktop\Legal_AI_Assistant\supabase\migrations\20260226_rag_init.sql`
- [x] T004 [P] Create migration to add `vector_status` and `last_vectorized` to `documents` table in `C:\Users\USER\Desktop\Legal_AI_Assistant\supabase\migrations\20260226_doc_status.sql`
- [x] T005 Implement `mixedbread-ai` SDK initialization in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\ai\mixedbread.ts`
- [x] T006 [P] Implement text splitting utility (500/50 config) in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\utils	ext-splitter.ts`
- [x] T007 Implement `processDocument` (chunking + embedding storage) in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\ai\vector-service.ts`

## Phase 3: [US1] Semantic Document Search

Story Goal: Allow lawyers to find relevant passages using natural language search
Independent Test: Searching for "termination" returns passages containing "ending a contract" from an uploaded document.

- [x] T008 [US1] Create `retrieve_context` SQL RPC function with similarity threshold (0.7) in `C:\Users\USER\Desktop\Legal_AI_Assistant\supabase\migrations\20260226_retrieval_rpc.sql`
- [x] T009 [US1] Implement retrieval wrapper logic in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\ai\vector-service.ts`
- [x] T010 [P] [US1] Create search interface component in `C:\Users\USER\Desktop\Legal_AI_Assistant\components\clients\vault-search.tsx`
- [x] T011 [US1] Integrate search results display into `C:\Users\USER\Desktop\Legal_AI_Assistant\app\(lawyer)\clients\[id]\vault\page.tsx`

## Phase 4: [US2] Cross-Client Data Isolation

Story Goal: Ensure client data confidentiality via strict vector namespace isolation
Independent Test: Search for Client A returns 0 results from Client B's documents.

- [x] T012 [US2] Implement Supabase RLS policies for `embeddings` table in `C:\Users\USER\Desktop\Legal_AI_Assistant\supabase\migrations\20260226_embeddings_rls.sql`
- [x] T013 [US2] Verify `retrieve_context` RPC strictly enforces `client_id` filtering in `C:\Users\USER\Desktop\Legal_AI_Assistant\supabase\migrations\20260226_retrieval_rpc.sql`

## Phase 5: [US3] Automated Playbook Integration

Story Goal: Automatically vectorize Playbook rules and include them in global context
Independent Test: Uploaded Playbook rules appear in all client search results.

- [x] T014 [US3] Update Playbook upload server action to trigger vectorization in `C:\Users\USER\Desktop\Legal_AI_Assistant\app\(admin)\admin\playbook\actions.ts`
- [x] T015 [US3] Ensure `retrieve_context` logic includes `client_id IS NULL` results in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\ai\vector-service.ts`

## Final Phase: Polish & Cross-cutting

Story Goal: Provide visual feedback and robust error handling for vectorization
Independent Test: Document vault shows "Vectorizing" status until ready; API retries on rate limits.

- [x] T016 Create `VectorStatusBadge` component in `C:\Users\USER\Desktop\Legal_AI_Assistant\components\ui\vector-status-badge.tsx`
- [x] T017 [P] Update Client Vault UI to display status badges in `C:\Users\USER\Desktop\Legal_AI_Assistant\app\(lawyer)\clients\[id]\vault\page.tsx`
- [x] T018 [P] Update Admin Playbook UI to display status badges in `C:\Users\USER\Desktop\Legal_AI_Assistant\app\(admin)\admin\playbook\page.tsx`
- [x] T019 Implement exponential backoff for Mixedbread API calls in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\ai\vector-service.ts`
- [x] T020 Implement `deleteDocumentVectors` trigger or service call in `C:\Users\USER\Desktop\Legal_AI_Assistant\lib\ai\vector-service.ts`
