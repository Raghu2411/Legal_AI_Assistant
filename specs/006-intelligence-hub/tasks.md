# Tasks: Intelligence Hub

**Input**: Design documents from `/specs/006-intelligence-hub/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: NONE. Automated tests are FORBIDDEN for this phase as per Constitution Principle III.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 [P] Install Vercel AI SDK and Groq dependencies in `package.json`
- [X] T002 [P] Create directory structure for intelligence-hub in `app/(lawyer)/intelligence-hub/` and `components/intelligence-hub/`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [X] T003 [P] Add `is_vendor` boolean field to `documents` table via migration in `supabase/migrations/`
- [X] T004 Create pgvector retrieval RPC with metadata filtering in `supabase/migrations/`
- [X] T005 [P] Implement vector retrieval helper with metadata filtering in `lib/supabase/vector-queries.ts`
- [X] T006 [P] Define stateless briefing templates in `lib/ai/briefing-templates.ts`
- [X] T007 [P] Initialize main Hub tab layout with shadcn/ui Tabs in `app/(lawyer)/intelligence-hub/page.tsx`

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Client Intelligence Chat (Priority: P1) 🎯 MVP

**Goal**: Enable natural language chat with document vault including session memory and citations.

**Independent Test**: Verify chat responses include numbered footnotes that link to document segments.

- [X] T008 [P] [US1] Create citation badge component in `components/intelligence-hub/citation-badge.tsx`
- [X] T009 [US1] Implement chat stream logic using Vercel AI SDK and Llama 3.3 in `lib/ai/chat-stream.ts`
- [X] T010 [US1] Create chat panel component with volatile in-memory state in `components/intelligence-hub/chat-panel.tsx`
- [X] T011 [US1] Integrate chat panel into main Hub page in `app/(lawyer)/intelligence-hub/page.tsx`
- [X] T012 [US1] Implement footnote interaction to highlight source text in `components/intelligence-hub/chat-panel.tsx`

**Checkpoint**: User Story 1 (Chat) is fully functional and testable independently.

---

## Phase 4: User Story 2 - Dynamic Executive Briefings (Priority: P2)

**Goal**: Adaptive document summaries based on document type.

**Independent Test**: Switch between document types and verify briefing sections adapt (e.g., Contract vs Evidence).

- [X] T013 [US2] Implement server action for on-demand briefing generation in `app/(lawyer)/intelligence-hub/actions.ts`
- [X] T014 [US2] Create briefing panel component with dynamic template rendering in `components/intelligence-hub/briefing-panel.tsx`
- [X] T015 [US2] Integrate briefing panel into main Hub page in `app/(lawyer)/intelligence-hub/page.tsx`

**Checkpoint**: User Story 2 (Briefings) is fully functional and testable independently.

---

## Phase 5: User Story 3 - Vendor Intelligence Filter (Priority: P3)

**Goal**: Isolate retrieval to vendor-related documents via metadata filter.

**Independent Test**: Toggle Vendor Mode and verify search results only include vendor documents.

- [X] T016 [US3] Add Vendor Document toggle to document upload form in `components/clients/upload-form.tsx`
- [X] T017 [US3] Create vendor mode toggle component for Hub in `components/intelligence-hub/vendor-toggle.tsx`
- [X] T018 [US3] Integrate vendor filter into retrieval service in `lib/ai/vector-service.ts`
- [X] T019 [US3] Add Vendor Mode content to main Hub page in `app/(lawyer)/intelligence-hub/page.tsx`

**Checkpoint**: User Story 3 (Vendor Mode) is fully functional and testable independently.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements and validation.

- [X] T020 [P] Documentation updates in `specs/006-intelligence-hub/`
- [X] T021 Perform final manual verification of all Success Criteria (SC-001 to SC-005) per `quickstart.md`

---

## Dependencies & Execution Order

1. **Setup (Phase 1)**: Must be completed first.
2. **Foundational (Phase 2)**: Depends on Setup; blocks all User Stories.
3. **User Story 1 (P1)**: High priority; can start after Phase 2.
4. **User Story 2 (P2)**: Medium priority; can start after Phase 2.
5. **User Story 3 (P3)**: Low priority; can start after Phase 2.
6. **Polish (Phase 6)**: Final step after all stories are implemented.

## Parallel Opportunities

- T001, T002 can run in parallel.
- T003, T005, T006, T007 can run in parallel (Foundational).
- Once Phase 2 is complete, US1, US2, and US3 can be implemented in parallel.
- T020 can run in parallel with other final tasks.
