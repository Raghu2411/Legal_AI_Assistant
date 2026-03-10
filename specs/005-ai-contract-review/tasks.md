---
description: "Task list for AI Contract Review implementation"
---

# Tasks: AI Contract Review (Step 5)

**Input**: Design documents from `/specs/005-ai-contract-review/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/actions.md

**Tests**: FORBIDDEN per Constitution Principle III. All verification must be performed manually according to the "Independent Test" criteria for each story.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Exact file paths are included in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Create Review Studio directory structure in `app/(lawyer)/review/[documentId]/`, `components/review/`, and `lib/review/`
- [x] T002 Install feature dependencies: `groq-sdk`, `@tiptap/react`, `@tiptap/starter-kit`, `lucide-react`, `react-pdf`, `docx`
- [x] T003 [P] Configure Groq client and environment variables in `lib/ai/groq-client.ts`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T004 Create Supabase migrations for `risk_analyses`, `clause_analyses`, and `golden_rules` tables in `supabase/migrations/`
- [x] T005 [P] Define Zod schemas for structured AI JSON output in `lib/review/schemas.ts`
- [x] T006 [P] Implement AI response parser with error handling in `lib/ai/parser.ts`
- [x] T007 [P] Create base AI review prompt template in `lib/ai/review-prompt.ts`
- [x] T008 [P] Define Server Action skeletons for `scanDocument`, `acceptRewrite`, `overrideRiskStatus`, and `markAsReviewed` in `lib/review/actions.ts`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Instant Contract Risk Analysis (Priority: P1) 🎯 MVP

**Goal**: Automatically analyze a contract upon entering the Review Studio and display risks using a Traffic Light System.

**Independent Test**: Open a document in Review Studio and verify that a full-document scan triggers automatically, showing a loading state and then populating the risk list with Green/Yellow/Red statuses.

### Implementation for User Story 1

- [x] T009 [P] [US1] Create three-pane layout shell in `components/review/layout/three-pane-layout.tsx`
- [x] T010 [P] [US1] Create document pane container in `components/review/layout/document-pane.tsx`
- [x] T011 [P] [US1] Create risk list pane with status filtering in `components/review/layout/risk-list-pane.tsx`
- [x] T012 [P] [US1] Implement risk item component with Traffic Light indicators in `components/review/risk-item.tsx`
- [x] T013 [US1] Implement `scanDocument` server action logic (fetch content + Groq call + persist) in `lib/review/actions.ts`
- [x] T014 [US1] Create Review Studio entry page in `app/(lawyer)/review/[documentId]/page.tsx`
- [x] T015 [US1] Implement automatic scan trigger using `useEffect` or Server Component logic in `app/(lawyer)/review/[documentId]/page.tsx`
- [x] T016 [US1] Integrate risk list display with scan results in `components/review/layout/risk-list-pane.tsx`

**Checkpoint**: User Story 1 is functional - documents are scanned and risks are listed automatically.

---

## Phase 4: User Story 2 - Side-by-Side Redlining & Immediate Update (Priority: P2)

**Goal**: Compare suggested AI changes side-by-side and apply them directly to a rich text editor.

**Independent Test**: Click 'View Suggestion' on a risk, verify the side-by-side modal opens, click 'Accept & Replace', and confirm the document text updates immediately in the editor.

### Implementation for User Story 2

- [x] T017 [P] [US2] Integrate TipTap editor with base extensions in `components/review/editor/tiptap-editor.tsx`
- [x] T018 [P] [US2] Create side-by-side comparison modal in `components/review/redline-modal.tsx`
- [x] T019 [P] [US2] Create action area pane for detailed risk view in `components/review/layout/action-pane.tsx`
- [x] T020 [US2] Implement `acceptRewrite` server action logic in `lib/review/actions.ts`
- [x] T021 [US2] Connect 'View Suggestion' trigger to redline modal in `components/review/risk-item.tsx`
- [x] T022 [US2] Implement 'Accept & Replace' logic to update TipTap state and call `acceptRewrite` action
- [x] T023 [US2] Implement document state persistence for manual edits in `components/review/editor/tiptap-editor.tsx`

**Checkpoint**: User Story 2 is functional - lawyers can accept AI rewrites and manually edit the contract.

---

## Phase 5: User Story 3 - Gap Analysis for Missing Clauses (Priority: P3)

**Goal**: Identify and flag missing mandatory clauses based on Golden Rules and Legal Playbook.

**Independent Test**: Use a document missing a "Mandatory" clause (e.g., Termination), run a scan, and verify a 'Red' risk appears explicitly labeled as a "Missing Clause" or "Gap."

### Implementation for User Story 3

- [x] T024 [P] [US3] Update `lib/ai/review-prompt.ts` to include gap analysis instructions and mandatory clause checklist
- [x] T025 [US3] Modify `scanDocument` in `lib/review/actions.ts` to process `is_gap` field from AI response
- [x] T026 [US3] Update `risk-list-pane.tsx` and `risk-item.tsx` to handle visual styling for gap analysis risks

**Checkpoint**: User Story 3 is functional - missing clauses are identified and flagged.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements, overrides, and final workflow completion.

- [x] T027 [P] Implement `overrideRiskStatus` server action logic with mandatory rationale in `lib/review/actions.ts`
- [x] T028 [P] Implement `markAsReviewed` server action logic to finalize document status in `lib/review/actions.ts`
- [x] T029 [P] Add manual status override UI and rationale form in `components/review/layout/action-pane.tsx`
- [x] T030 Add concurrent edit warning alert using presence detection in `app/(lawyer)/review/[documentId]/page.tsx`
- [x] T034 [US1] Implement "Scan History" dropdown/switcher to view and navigate previous RiskAnalyses in `components/review/layout/risk-list-pane.tsx`
- [x] T031 Implement PDF and DOCX export functionality in `app/(lawyer)/review/[documentId]/page.tsx`
- [x] T032 Add global loading overlays and optimistic UI updates for scan/accept actions
- [x] T033 [P] Final documentation and quickstart validation check

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately.
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories.
- **User Stories (Phase 3+)**: All depend on Foundational completion.
    - US1 (P1) is the MVP and should be completed first.
    - US2 and US3 can proceed in parallel once US1's layout is stable.
- **Polish (Phase 6)**: Depends on at least US1 and US2 completion.

### Parallel Opportunities

- T003 (Setup) can run in parallel with T001/T002.
- T005, T006, T007, T008 (Foundational) can run in parallel.
- T009, T010, T011, T012 (US1 UI components) can run in parallel.
- T017, T018, T019 (US2 UI components) can run in parallel.
- T024 (US3 Prompt) can be worked on while US1/US2 are in progress.
- T027, T028 (Server Actions) can run in parallel.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 & 2.
2. Complete Phase 3 (US1).
3. **VALIDATE**: Manually verify that opening a document triggers a scan and populates the risk list.

### Incremental Delivery

1. Foundation -> Scan & List (US1) -> Redlining (US2) -> Gap Analysis (US3) -> Polish.
2. Each phase delivers a testable increment to the "Review Studio."
