# Implementation Plan: Smart Triage & Operations

**Branch**: `008-smart-triage-operations` | **Date**: 2026-03-22 | **Spec**: [specs/008-smart-triage-operations/spec.md]
**Input**: Feature specification from `/specs/008-smart-triage-operations/spec.md`

## Summary
The goal of Step 8 is to implement a high-efficiency Triage Queue and Operations Dashboard. We will leverage Groq/Llama 3.3 to classify incoming documents based on Admin "Golden Rules" (Step 5) and automatically extract legal obligations (dates, milestones). These obligations will follow a 'Pending' to 'Confirmed' state transition (Principle XXXII), with confirmed items plotted on a shadcn/ui Calendar. Dual-scope compliance (firm policy vs. regulatory) will be flagged in real-time.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router), Python 3.11 (FastAPI)
**Primary Dependencies**: shadcn/ui, Groq SDK (Llama 3.3), Supabase, Vercel AI SDK
**Storage**: Supabase (PostgreSQL, pgvector)
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web
**Project Type**: Web Application
**Performance Goals**: Triage classification under 10 seconds for <20 pages.
**Constraints**: All AI-extracted data MUST remain 'Pending' until confirmed by a lawyer.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle III (No Testing)**: PASSED. No automated tests will be included.
- **Principle XXXI (Golden Rule Triage)**: PASSED. Logic uses Admin-defined rules to classify Standard vs. Complex.
- **Principle XXXII (Obligation Verification)**: PASSED. Confirmation workflow ('Pending' status) implemented in `obligations` table.
- **Principle XXXIII (Dual-Layer Compliance)**: PASSED. Multi-shot AI prompting covers both policy and regulatory layers.
- **Principle XXXIV (Operational Auditability)**: PASSED. All confirmations and triage decisions logged in `activity_logs`.

## Project Structure

### Documentation (this feature)

```text
specs/008-smart-triage-operations/
├── plan.md              # This file
├── research.md          # Implementation decisions
├── data-model.md        # Obligations schema
├── quickstart.md        # Verification steps
├── contracts/           # API schemas
└── tasks.md             # Implementation tasks
```

### Source Code (repository root)

```text
app/(lawyer)/dashboard/operations/
├── layout.tsx
├── page.tsx             # Main dashboard
├── calendar/            # Calendar view
└── triage/              # Triage Queue view

components/operations/
├── triage-table.tsx
├── calendar-view.tsx
├── obligation-item.tsx
└── compliance-sidebar.tsx

lib/ai/
├── triage-service.ts    # Groq classification logic
└── extractor.ts         # Obligation extraction logic

supabase/migrations/
└── 20260322_obligations_schema.sql
```

**Structure Decision**: Option 2: Web application. The logic is split between `lib/ai` (backend processing) and `components/operations` (frontend display).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A | | |
