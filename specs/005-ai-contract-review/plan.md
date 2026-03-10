# Implementation Plan: AI Contract Review (Step 5)

**Branch**: `005-ai-contract-review` | **Date**: 2026-03-09 | **Spec**: [specs/005-ai-contract-review/spec.md](spec.md)

## Summary

Implement the "Review Studio," a three-pane interactive interface for lawyers to perform AI-assisted contract analysis. The system will use Groq (Llama-3.3-70b) to scan documents against a combined context of Step 4 RAG (Legal Playbook) and Admin-defined "Golden Rules." Results will be displayed via a Traffic Light System (Red/Yellow/Green) with side-by-side redlining and a rich text editor (TipTap) for immediate state updates and manual refinements.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router)
**Primary Dependencies**: Groq SDK, TipTap (Editor), shadcn/ui (Three-pane layout), Lucide React (Icons), Supabase (Storage/DB), react-pdf, docx
**Storage**: Supabase PostgreSQL (`risk_analyses`, `clause_analyses`, `golden_rules`), Supabase Storage (PDF/DOCX)
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web (Desktop optimized for Review Studio)
**Project Type**: Web Application (Feature addition)
**Performance Goals**: AI Analysis results in <15s; UI state updates <100ms
**Constraints**: 1024 vector dimensions (mxbai-embed-large-v1); Last Writer Wins for concurrent edits
**Scale/Scope**: Lawyers reviewing 1-50 page contracts; high-density JSON structured output from LLM

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle III (No Testing)**: Confirmed. No automated tests will be created.
- **Principle XVII (Instant Scan)**: Plan includes a `useEffect` or Server Action trigger upon Review Studio entry.
- **Principle XVIII (Traffic Light)**: Data model and UI components explicitly support Red/Yellow/Green statuses.
- **Principle XIX (Golden Rule Precedence)**: Prompt engineering will explicitly instruct the LLM to prioritize Golden Rules context.
- **Principle XX (Mandatory Clauses)**: AI review prompt will include a "Gap Analysis" task to identify missing requirements.
- **Principle XXI (Side-by-Side/Immediate Update)**: TipTap integration and custom modal support side-by-side redlines with immediate state persistence.

## Project Structure

### Documentation (this feature)

```text
specs/005-ai-contract-review/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── actions.md       # Server Action definitions
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
app/
├── (lawyer)/
│   └── review/
│       └── [documentId]/
│           └── page.tsx        # Review Studio Entry
components/
├── review/
│   ├── layout/
│   │   ├── three-pane-layout.tsx
│   │   ├── document-pane.tsx
│   │   ├── risk-list-pane.tsx
│   │   └── action-pane.tsx
│   ├── editor/
│   │   └── tiptap-editor.tsx
│   ├── risk-item.tsx
│   └── redline-modal.tsx
lib/
├── ai/
│   ├── groq-client.ts
│   ├── review-prompt.ts
│   └── parser.ts
├── review/
│   ├── actions.ts             # Server Actions (scan, accept, override)
│   └── schemas.ts             # Zod schemas for AI JSON output
```

**Structure Decision**: Integrated within the existing Next.js App Router structure under a new `(lawyer)/review` route to maintain separation of concerns while leveraging shared UI components.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None      | N/A        | N/A                                 |
