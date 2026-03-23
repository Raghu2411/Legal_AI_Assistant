# Tasks: Policy Evolution Studio

**Input**: Design documents from `/specs/009-policy-evolution/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are FORBIDDEN for this phase as per Constitution Principle III. DO NOT include any automated tests.

## Phase 1: Setup (Shared Infrastructure)

- [x] T001 Create Evolution Studio directory structure: `app/(admin)/admin/evolution/`, `api/evolution/`, `api/generation/`
- [x] T002 Install `python-docx` in the FastAPI backend environment
- [x] T003 Configure `groq` environment variables for gap analysis logic

## Phase 2: Foundational (Blocking Prerequisites)

- [x] T004 Create migration for `compliance_standards`, `policy_suggestions`, and `version_history` tables
- [x] T005 Update `playbooks` and `golden_rules` tables with `version` and `last_updated` fields
- [x] T006 Implement RLS policies for new tables (admin-only access)
- [x] T007 Create base `evolution-queries.ts` for database operations and history logging

## Phase 3: User Story 1 - Compliance Auditing & Suggestions (Priority: P1)

- [x] T008 [P] [US1] Implement `ComplianceStandard` upload handler in `api/evolution/audit/route.ts`
- [x] T009 [P] [US1] Create Gap Analysis prompts in `lib/ai/evolution-prompts.ts`
- [x] T010 [US1] Implement Gap Analysis orchestrator in `lib/ai/evolution-orchestrator.ts` (Stage 1: Retrieval, Stage 2: Comparison)
- [x] T011 [US1] Implement `GET /api/evolution/suggestions/{job_id}` endpoint

## Phase 4: User Story 2 - Evolution Studio Review (Priority: P1) 🎯 MVP

- [x] T012 [P] [US2] Build `EvolutionStudio` dashboard in `app/(admin)/admin/evolution/page.tsx`
- [x] T013 [P] [US2] Create `AuditPanel` component for standard uploads and status tracking
- [x] T014 [P] [US2] Implement `SuggestionsList` with side-by-side view and checkboxes
- [x] T015 [US2] Implement `POST /api/evolution/approve` endpoint for selective approval

## Phase 5: User Story 3 - Playbook Generation & RAG Sync (Priority: P2)

- [x] T016 [US3] Implement server-side DOCX generator using `python-docx` in `lib/operations/docx-generator.py`
- [x] T017 [US3] Implement `POST /api/generation/refresh-playbook` endpoint
- [x] T018 [US3] Integrate `processDocument` trigger after successful playbook generation
- [x] T019 [US3] Add status polling endpoint `GET /api/generation/playbook-status/{playbook_id}`

## Phase 6: User Story 4 - Instant Golden Rule Propagation (Priority: P1)

- [x] T020 [US4] Update Golden Rule approval logic to ensure atomic database commits

## Phase 7: User Story 5 - Accountability & Version History (Priority: P2)

- [x] T023 [P] [US5] Build `HistoryView` component for the audit trail
- [x] T024 [P] [US5] Implement `GET /api/evolution/history` endpoint with "Before vs After" diff logic
- [x] T025 [US5] Implement `POST /api/evolution/rollback` endpoint with mandatory version increment and RAG sync

## Phase 8: Polish & Cross-Cutting Concerns

- [x] T026 Add 'Last Write Wins' concurrency notifications in the Evolution Studio UI
- [x] T027 Finalize and document the `version_history` schema and rollback procedures in `docs/`
- [x] T028 Perform a final walkthrough of the Evolution Studio as an Admin to verify end-to-end functionality
