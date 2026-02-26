# Tasks: Client & Case Management

**Input**: Design documents from `/specs/003-client-management/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: NO automated tests are included as per Constitution Principle III (DEBT-BY-DESIGN). Manual verification steps are defined in `quickstart.md`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 [P] Create directory structure for clients in `app/`, `components/`, and `lib/` per implementation plan
- [X] T002 [P] Create `client-vaults` private storage bucket in Supabase dashboard and configure restricted file types (PDF, DOCX, TXT)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T003 Setup database schema for `clients` and `documents` tables in Supabase SQL Editor per `data-model.md`
- [X] T004 Implement `generate_client_case_id` PostgreSQL function and trigger in Supabase SQL Editor per `data-model.md`
- [X] T005 [P] Enable Row-Level Security (RLS) and apply policies for `clients`, `documents`, and `storage.objects` per `data-model.md`

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Lawyer Client Onboarding (Priority: P1) 🎯 MVP

**Goal**: Enable Lawyers to add clients with automated Case ID generation and view their private portfolio.

**Independent Test**: Sign in as Lawyer -> Add Client via `/clients/new` -> Verify client appears in `/clients` with `lawyerName-XXXX` ID.

### Implementation for User Story 1

- [X] T006 [P] [US1] Define Zod schema for client onboarding in `lib/clients/actions.ts`
- [X] T007 [US1] Implement `createClient` server action in `lib/clients/actions.ts`
- [X] T008 [P] [US1] Create `client-form.tsx` component using shadcn/ui and react-hook-form in `components/clients/`
- [X] T009 [US1] Create client onboarding page in `app/(lawyer)/clients/new/page.tsx`
- [X] T010 [P] [US1] Create `client-table.tsx` component for list views in `components/clients/`
- [X] T011 [US1] Create lawyer client list page in `app/(lawyer)/clients/page.tsx`

**Checkpoint**: User Story 1 is fully functional. Lawyers can onboard and view their clients.

---

## Phase 4: User Story 2 - Document Vault Management (Priority: P2)

**Goal**: Enable Lawyers to upload, categorize, and manage client documents in a secure vault.

**Independent Test**: Open Client Vault -> Upload PDF with "Contract" category -> Verify display in vault and storage path in Supabase.

### Implementation for User Story 2

- [X] T012 [P] [US2] Implement `uploadDocument` and `deleteDocument` server actions in `lib/clients/actions.ts`
- [X] T013 [P] [US2] Create `upload-form.tsx` component with document type dropdown in `components/clients/`
- [X] T014 [US2] Create `vault-view.tsx` component to list and manage documents in `components/clients/`
- [X] T015 [P] [US2] Create client overview page in `app/(lawyer)/clients/[id]/page.tsx`
- [X] T016 [US2] Create document vault page in `app/(lawyer)/clients/[id]/vault/page.tsx`

**Checkpoint**: User Stories 1 and 2 are functional. Lawyers can manage clients and their document vaults.

---

## Phase 5: User Story 3 - Admin Firm-Wide Oversight (Priority: P3)

**Goal**: Provide Admins with a firm-wide searchable client list and quality control capabilities.

**Independent Test**: Sign in as Admin -> Access `/admin/clients` -> Search by Client and Lawyer name -> Edit client details via modal.

### Implementation for User Story 3

- [X] T017 [P] [US3] Implement `getFirmClients` and `updateClient` server actions in `lib/clients/actions.ts`
- [X] T018 [US3] Update `client-table.tsx` to support Admin-specific columns (Lawyer Name) and actions (Edit)
- [X] T019 [P] [US3] Create `client-edit-modal.tsx` component for Admin quality control in `components/clients/`
- [X] T020 [US3] Create admin firm-wide clients page in `app/(admin)/admin/clients/page.tsx` with server-side search
- [X] T020.1 [US3] Update Admin Dashboard 'Recent Activity' metrics to reflect 24-hour window in `app/(admin)/admin/page.tsx` (FR-011)

**Checkpoint**: All user stories are functional. Admins have oversight and edit capabilities.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements and system integration

- [X] T021 [P] Integrate audit logging for client creation and document uploads in `lib/clients/actions.ts` using existing `logs` table
- [X] T022 [P] Add empty state views with "Add Client" CTA for lawyer client list per `spec.md`
- [X] T023 Run full `quickstart.md` validation flow to ensure compliance with all requirements

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: Can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (T001-T002). Blocks all User Stories.
- **User Stories (Phase 3-5)**: All depend on Foundational completion.
  - US1 is the MVP and should be completed first.
  - US2 can proceed in parallel with US1 if data models are ready, but ideally follows US1 to use the created clients.
  - US3 can proceed in parallel once US1 provides data to view.

### Parallel Opportunities

- T001 and T002 can run in parallel.
- T005 can run in parallel with T003/T004 once tables are created.
- Once Phase 2 is complete, US1, US2, and US3 can be worked on in parallel by different developers.
- Within stories, UI components ([P]) and Server Actions ([P]) can often be developed simultaneously.

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 & 2 (Setup & Foundation).
2. Complete Phase 3 (US1: Client Onboarding).
3. **Validate**: Verify Case ID generation and private list visibility.

### Incremental Delivery

1. Deliver US1 (Onboarding) -> Foundation for everything else.
2. Deliver US2 (Vault) -> Adds document management value.
3. Deliver US3 (Admin) -> Adds oversight and management capabilities.
