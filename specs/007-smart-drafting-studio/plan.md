# Implementation Plan: Smart Drafting Studio

**Branch**: `007-smart-drafting-studio` | **Date**: 2026-03-22 | **Spec**: `/specs/007-smart-drafting-studio/spec.md`
**Input**: Feature specification from `/specs/007-smart-drafting-studio/spec.md`

## Summary
The Smart Drafting Studio is an interactive, AI-driven document creation environment. It features a two-column layout with a chat-based "Interview Orchestrator" (powered by Groq/Llama 3.3) and a Tiptap rich text editor. The system guides lawyers through a document-specific interview, generating and inserting clauses in real-time while maintaining manual override sovereignty. Final documents are stored in Supabase and automatically indexed for RAG.

## Technical Context
**Language/Version**: TypeScript (Next.js 14 App Router), Python 3.11 (FastAPI)
**Primary Dependencies**: Tiptap, shadcn/ui, Groq SDK (Llama 3.3), Vercel AI SDK, Supabase (Auth, DB, Storage, Vector)
**Storage**: Supabase PostgreSQL (pgvector), Supabase Storage (client-documents bucket)
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web (Responsive Desktop Optimized)
**Project Type**: Web Application
**Performance Goals**: AI fragments in < 2s, initiation in < 5s
**Constraints**: In-memory session only (MVP), RLS isolation, No automated tests
**Scale/Scope**: Lawyers drafting complex legal documents (NDAs, Contracts, etc.)

## Constitution Check

| Principle | Status | Summary/Justification |
|-----------|--------|-----------------------|
| III. No Testing | ✅ PASS | No automated tests will be implemented; manual verification only. |
| IV. Data Isolation via RLS | ✅ PASS | All drafting sessions and document storage will be gated by RLS policies. |
| XXVI. Mandatory Client Context | ✅ PASS | Client selection is a hard prerequisite for starting a draft. |
| XXVII. Interactive Interview Drafting | ✅ PASS | Core architecture uses an interview-based flow. |
| XXVIII. Manual Override Sovereignty | ✅ PASS | Tiptap editor allows full manual editing at any time. |
| XXIX. Automatic RAG Indexing | ✅ PASS | Final save triggers the RAG ingestion pipeline. |
| XXX. Explicit Email Generation | ✅ PASS | Email generation is a manual user-triggered action. |

## Project Structure

### Documentation (this feature)

```text
specs/007-smart-drafting-studio/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
app/
└── (lawyer)/
    └── dashboard/
        └── drafting/
            ├── page.tsx            # Drafting Studio UI
            └── layout.tsx
components/
└── drafting/
    ├── chat-panel.tsx              # Interview interface
    ├── editor-panel.tsx            # Tiptap integration
    ├── client-selector.tsx         # Pre-drafting gate
    └── email-modal.tsx             # Draft Email utility
lib/
└── ai/
    ├── drafting-orchestrator.ts    # Groq state management
    └── drafting-prompts.ts         # Document-specific templates
```

**Structure Decision**: Single project web application structure as established in the current repo.

## Complexity Tracking
> No violations of the constitution detected.
