# Implementation Plan: RAG Infrastructure

**Branch**: `004-rag-infrastructure` | **Date**: 2026-02-26 | **Spec**: [specs/004-rag-infrastructure/spec.md](C:\Users\USER\Desktop\Legal_AI_Assistant\specs\004-rag-infrastructure\spec.md)
**Input**: Feature specification from `/specs/004-rag-infrastructure/spec.md`

## Summary
Implement a semantic search (RAG) system for legal documents using Supabase (PostgreSQL/pgvector) and Mixedbread AI. The system will automatically chunk, vectorize, and store embeddings for uploaded documents (lawyer-client files and global playbooks) with strict namespace isolation and asynchronous status tracking.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router)
**Primary Dependencies**: `mixedbread-ai` SDK, `langchain` (RecursiveCharacterTextSplitter), `pdf-parse`, `mammoth` (for DOCX)
**Storage**: Supabase PostgreSQL (pgvector), Supabase Storage
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web (Next.js Edge/Server Actions)
**Project Type**: Web Application (SaaS)
**Performance Goals**: Retrieval < 2s, Vectorization < 30s per standard doc
**Constraints**: 1024-dimension vectors, 0.7 similarity threshold, 500/50 chunking
**Scale/Scope**: Multitenant (lawyers/clients), strict RLS enforcement

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Implementation Strategy |
|-----------|--------|-------------------------|
| I. Clean Code | ✅ | Design uses flat service structure in `lib/ai` and SQL RPC for retrieval. |
| III. No Testing | ✅ | All verification steps in `quickstart.md` are manual. |
| IV. Data Isolation | ✅ | RLS policies for `embeddings` defined in `rag_init.sql` migration. |
| X. Single Embedding Model | ✅ | `mixedbread-ai` SDK configured for `mxbai-embed-large-v1`. |
| XI. Namespace Isolation | ✅ | `retrieve_context` SQL function includes `client_id` OR NULL filter. |
| XII. Query Prefix | ✅ | Prefix handled in `vector-service.ts` retrieval method. |
| XIII. Lifecycle Sync | ✅ | Handled via server actions and `ON DELETE CASCADE` on `embeddings`. |
| XIV. Auth-Gated | ✅ | Role-based gate implemented in vectorization server actions. |
| XV. Failure Atomicity | ✅ | 'Error' status transition used for tracking and cleanup. |
| XVI. Idempotency | ✅ | `delete before insert` logic embedded in `processDocument`. |


## Project Structure

### Documentation (this feature)

```text
specs/004-rag-infrastructure/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (not yet created)
```

### Source Code (repository root)

```text
app/
├── (admin)/admin/playbook/   # Updated for status visibility
├── (lawyer)/clients/[id]/vault/ # Updated for status visibility
└── api/vectorize/           # Edge function/Route handler for async processing

lib/
├── ai/
│   ├── mixedbread.ts        # SDK initialization
│   └── vector-service.ts    # processDocument & retrieval logic
├── supabase/
│   └── admin.ts             # Service role client for triggers
└── utils/
    └── text-splitter.ts     # Chunking logic (500/50)

supabase/
└── migrations/
    └── [timestamp]_rag_init.sql # pgvector, table, and HNSW index
```

**Structure Decision**: Integrated into existing Next.js App Router structure with new AI service layer in `lib/ai`.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No violations detected.*
