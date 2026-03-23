# Implementation Plan: Policy Evolution Studio

**Branch**: `009-policy-evolution` | **Date**: 2026-03-22 | **Spec**: [specs/009-policy-evolution/spec.md]
**Input**: Feature specification from `/specs/009-policy-evolution/spec.md`

## Summary
Implement the **Evolution Studio** for Admins to manage and evolve firm legal logic. This includes an AI-driven **Gap Analysis** engine to audit internal rules against external standards, a server-side **DOCX generator** for Playbook updates, and an immutable **Version History** for full accountability. Key integration points include ensuring Golden Rule updates are instantly propagated to Triage (Module 2) and Review (Module 5) prompts.

## Technical Context
**Language/Version**: Python 3.11 (FastAPI), TypeScript (Next.js 14 App Router)
**Primary Dependencies**: `python-docx` (for server-side generation), `groq` (Llama 3.3 for audit), `supabase-js`
**Storage**: Supabase (PostgreSQL), Supabase Storage (DOCX artifacts)
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web (Admin Dashboard)
**Project Type**: Web Application / AI Integration
**Performance Goals**: Gap Analysis < 30s for 50-page docs; Golden Rule Sync < 500ms
**Constraints**: strictly 'admin' role access; mandatory RAG re-indexing on Playbook updates; 1024-dim embeddings (mxbai-embed-large-v1)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Clean Code (I)**: ✅ Declarative API endpoints defined in contracts.
2. **No Testing (III)**: ✅ No automated tests planned.
3. **Data Isolation (IV)**: ✅ RLS enforced for 'admin' role.
4. **Auth-Gated Processing (XIV)**: ✅ 'admin' role mandatory for Evolution Studio.
5. **Golden Rule Precedence (XIX)**: ✅ Prompts fetch updated rules from DB on every run.
6. **Versioned Policy Evolution (XXXVI)**: ✅ Playbook updates trigger version increment and RAG sync.
7. **Instant Golden Rule Sync (XXXVII)**: ✅ Atomic DB updates for Golden Rules.
8. **Immutable Policy Audit Trail (XXXVIII)**: ✅ `version_history` table stores 'Before' and 'After' states.

## Project Structure
```text
app/
├── (admin)/
│   └── admin/
│       └── evolution/           # Evolution Studio UI
│           ├── page.tsx
│           ├── audit-panel.tsx
│           ├── suggestions-list.tsx
│           └── history-view.tsx
├── api/
│   ├── evolution/               # Policy evolution API
│   │   ├── audit/route.ts
│   │   ├── approve/route.ts
│   │   └── history/route.ts
│   └── generation/              # Document generation API
│       └── refresh/route.ts

lib/
├── ai/
│   ├── evolution-orchestrator.ts # Gap Analysis logic
│   └── evolution-prompts.ts      # LLM prompts for auditing
├── operations/
│   └── docx-generator.py        # Python script for DOCX (FastAPI)
└── supabase/
    └── evolution-queries.ts      # History and rollback logic
```

**Structure Decision**: Integrated within the existing Next.js App Router for UI/API and FastAPI backend for heavy Python-based DOCX generation.

## Complexity Tracking
> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | | |
