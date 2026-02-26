# Tasks: Admin CRUD Console

**Input**: Design documents from `/specs/002-admin-crud-console/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/ui-routes.md

**Tests**: Tests are FORBIDDEN for this phase as per Constitution Principle III. DO NOT include any automated tests.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)

## Phase 1: Setup

**Purpose**: Project initialization and basic structure

- [x] T001 Create admin route structure in app/(admin)/admin/layout.tsx and Sidebar component
- [x] T002 Install dependencies: groq-sdk, pdf-parse
- [x] T003 [P] Add shadcn/ui components: table, dialog, scroll-area, tabs

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T004 Create `logs` table in Supabase per data-model.md
- [x] T005 Create `playbooks` table in Supabase per data-model.md
- [x] T006 Create `playbooks` storage bucket in Supabase with Admin-only write policies
- [x] T007 [P] Implement `lib/supabase/admin.ts` with server-side Admin client and data reassignment logic
- [x] T008 Update `app/auth/actions.ts` to default all new sign-ups to 'lawyer' role and log user creation

## Phase 3: User Story 1 - User Oversight & Audit Logs (Priority: P1)

**Goal**: Centralized dashboard for user management and audit log visibility

**Independent Test**: Navigate to /admin/users and /admin/logs as Admin; verify data is listed and searchable.

### Implementation for User Story 1

- [x] T009 [P] [US1] Create User DataTable component in components/admin/user-table.tsx
- [x] T010 [US1] Implement user role toggle action in app/(admin)/admin/users/page.tsx
- [x] T010b [US1] Implement Lawyer Deletion/Archive action with mandatory data reassignment trigger in app/(admin)/admin/users/page.tsx
- [x] T011 [P] [US1] Create Log DataTable component in components/admin/log-table.tsx
- [x] T012 [US1] Build Audit Trail page with filtering in app/(admin)/admin/logs/page.tsx
- [x] T013 [US1] Implement Dashboard Overview with summary stats in app/(admin)/admin/page.tsx
- [x] T014 [US1] Add login/logout logging to lib/supabase/middleware.ts

## Phase 4: User Story 2 - Legal Playbook & Golden Rules Management (Priority: P1)

**Goal**: Management of firm-wide legal guidelines via hybrid storage

**Independent Test**: Upload a PDF and save Golden Rules; verify they appear in storage and DB.

### Implementation for User Story 2

- [x] T015 [P] [US2] Create hybrid Playbook form in components/admin/playbook-form.tsx
- [x] T016 [US2] Implement PDF parsing logic using pdf-parse in lib/playbook/parser.ts
- [x] T017 [US2] Build Playbook management page in app/(admin)/admin/playbook/page.tsx
- [x] T018 [US2] Implement version history list in app/(admin)/admin/playbook/page.tsx
- [x] T019 [US2] Add file upload and DB update logic to server actions in app/admin/playbook/actions.ts

## Phase 5: User Story 3 - AI Context Integration (Priority: P2)

**Goal**: AI response generation using Playbook and Golden Rules context

**Independent Test**: Ask the AI a question; verify it cites the Playbook or Golden Rules.

### Implementation for User Story 3

- [x] T020 [P] [US3] Create AI context retrieval utility in lib/ai/groq.ts
- [x] T021 [US3] Implement LLM-based conflict detection logic (prompting AI to identify contradictions between Golden Rules and Playbook) in lib/ai/groq.ts
- [x] T022 [US3] Update AI prompt template to include explicit source citations
- [x] T023 [US3] Integrate context retrieval into the main chat logic

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements and cleanup

- [x] T024 [P] Implement 90-day log retention via Supabase Edge Function with a daily CRON trigger
- [x] T025 [P] Finalize RLS policies for all new tables
- [x] T026 Run quickstart.md validation steps

## Dependencies & Execution Order

1. **Setup (Phase 1)**: Must be first.
2. **Foundational (Phase 2)**: Depends on Phase 1.
3. **User Story 1 & 2 (Phase 3 & 4)**: Can run in parallel after Phase 2 is complete.
4. **User Story 3 (Phase 5)**: Depends on Phase 4 (context sources must exist).
5. **Polish (Phase 6)**: Final step.
