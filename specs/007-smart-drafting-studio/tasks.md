# Tasks: Smart Drafting Studio

**Input**: Design documents from `/specs/007-smart-drafting-studio/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Automated tests are FORBIDDEN for this phase as per Constitution Principle III. All verification is manual per the "Independent Test" criteria in each user story.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup

**Purpose**: Project initialization and basic structure

- [X] T001 Create drafting dashboard directory structure in `app/(lawyer)/dashboard/drafting/`
- [X] T002 Create component directory for drafting in `components/drafting/`
- [X] T003 [P] Configure Tiptap dependencies and starter kit in `package.json`

---

## Phase 2: Foundational

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [X] T004 Create Supabase migration for `activity_logs` table and `documents` table extensions in `supabase/migrations/20260322_smart_drafting_init.sql`
- [X] T005 [P] Implement `drafting-prompts.ts` with initial NDA/Contract templates in `lib/ai//`
- [X] T006 [P] Create base layout for the drafting studio in `app/(lawyer)/dashboard/drafting/layout.tsx`
- [X] T007 Initialize Tiptap editor with custom extensions (for highlighting placeholders) in `components/drafting/editor-panel.tsx`

---

## Phase 3: User Story 1 - Client-Contextual Drafting Initiation (Priority: P1)

**Goal**: Select a client, document type, and name before starting a session.

**Independent Test**: Verify that the "Start Drafting" button is only active when all fields are filled and client details are loaded.

- [X] T008 [US1] Create client and document type selector component in `components/drafting/client-selector.tsx`
- [X] T009 [US1] Implement pre-drafting gate logic in `app/(lawyer)/dashboard/drafting/page.tsx`
- [X] T010 [P] [US1] Implement activity logging for 'DRAFTING_START' in `lib/ai/drafting-orchestrator.ts`

---

## Phase 4: User Story 2 - Interactive AI-Assisted Interview (Priority: P1)

**Goal**: AI-guided interview that builds the document progressively.

**Independent Test**: Verify that AI questions appear in chat and responses trigger document updates in the editor.

- [X] T011 [US2] Implement Groq-powered `drafting-orchestrator.ts` for stateful interview logic in `lib/ai/`
- [X] T012 [US2] Create chat interface component in `components/drafting/chat-panel.tsx`
- [X] T013 [US2] Create API route for drafting chat in `app/api/drafting/chat/route.ts`
- [X] T014 [US2] Implement Tiptap sync logic to handle `CLAUSE_UPDATE` events from the AI stream in `components/drafting/editor-panel.tsx`
- [X] T015 [US2] Implement `[MISSING_TERM]` placeholder insertion logic for skipped questions in `lib/ai/drafting-orchestrator.ts`

---

## Phase 5: User Story 3 - Manual Editor Sovereignty (Priority: P2)

**Goal**: Manual editing capability with AI-locking protection.

**Independent Test**: Verify manual edits are possible and that the editor locks while the AI is generating text.

- [X] T016 [US3] Implement editor locking state and UI feedback in `components/drafting/editor-panel.tsx`
- [X] T017 [US3] Ensure manual edits are preserved during AI context updates in `components/drafting/editor-panel.tsx`

---

## Phase 6: User Story 4 - Finalization, Storage & RAG Indexing (Priority: P2)

**Goal**: Save finalized document to Supabase and trigger RAG indexing.

**Independent Test**: Verify document appears in Vault and its content is searchable in Intelligence Hub.

- [X] T018 [US4] Implement "Finalize & Save" logic (PDF/DOCX generation) in `app/api/drafting/finalize/route.ts`
- [X] T019 [US4] Implement Supabase Storage upload to `client-documents` bucket in `app/api/drafting/finalize/route.ts`
- [X] T020 [US4] Integrate `processDocument` from `lib/ai/vector-service.ts` to trigger RAG indexing upon save.

---

## Phase 7: User Story 5 - Professional Cover Email Generation (Priority: P3)

**Goal**: Explicitly triggered cover email generation based on document content.

**Independent Test**: Verify email draft accurately reflects the finalized document and client context.

- [X] T021 [US5] Create email generation API in `app/api/drafting/email/route.ts`
- [X] T022 [US5] Implement email preview modal in `components/drafting/email-modal.tsx`
- [X] T023 [P] [US5] Log 'EMAIL_GENERATED' action in `activity_logs` table.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements and cleanup.

- [X] T024 [P] Refine UI responsiveness for mobile/tablet in `app/(lawyer)/dashboard/drafting/`
- [X] T025 Performance optimization for AI streaming and editor sync.
- [X] T026 Final validation of RLS policies for `activity_logs`.

---

## Dependencies & Execution Order

1. **Setup (Phase 1)**: Must be completed first.
2. **Foundational (Phase 2)**: Depends on Phase 1. BLOCKS all user stories.
3. **User Story 1 (P1)**: Depends on Phase 2. Enables the entry point for drafting.
4. **User Story 2 (P1)**: Depends on US1 completion. Core drafting logic.
5. **User Story 3 (P2)**: Depends on US2 completion. Extends editor behavior.
6. **User Story 4 (P2)**: Depends on US3 completion. Required for persistence.
7. **User Story 5 (P3)**: Depends on US4 completion. Final delivery utility.
8. **Polish (Final Phase)**: Runs after all stories are verified.

## Parallel Opportunities

- T003, T005, T006 can be done in parallel once T001 and T002 are ready.
- T010 can be done in parallel with T008 and T009.
- T023 can be done in parallel with T021 and T022.

---

## Implementation Strategy

1. **MVP First**: Complete US1 and US2 to have a functional (but non-persistent) drafting interview.
2. **Incremental Delivery**: Add persistence (US4), then manual overrides (US3), and finally the email utility (US5).
3. **Manual Validation**: Verify each user story independently before proceeding to the next phase.
