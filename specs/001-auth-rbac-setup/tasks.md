# Tasks: Auth & RBAC Setup

**Input**: Design documents from `/specs/001-auth-rbac-setup/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: The examples below include test tasks. Tests are FORBIDDEN for this phase as per Constitution Principle III. DO NOT include any automated tests (unit, integration, or e2e).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Initialize Next.js 14 project with App Router and Tailwind CSS
- [x] T002 [P] Install dependencies: @supabase/ssr, @supabase/supabase-js, lucide-react, shadcn/ui, react-hook-form, zod
- [x] T003 Initialize shadcn/ui and configure theme
- [x] T004 [P] Configure environment variables in .env.local (NEXT_PUBLIC_SUPABASE_URL, NEXT_PUBLIC_SUPABASE_ANON_KEY)
- [x] T005 [P] Setup Supabase client utilities in lib/supabase/client.ts and lib/supabase/server.ts

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [x] T006 Create 'profiles' table and 'user_role' enum in Supabase SQL Editor
- [x] T007 Setup SQL Trigger for auto-profile creation in Supabase
- [x] T008 [P] Enable Row-Level Security (RLS) and define policies for 'profiles' table
- [x] T009 [P] Implement Supabase Middleware in lib/supabase/middleware.ts for session management
- [x] T010 Implement main Next.js middleware in middleware.ts using lib/supabase/middleware.ts
- [x] T011 [P] Create shared UI components (Button, Input, Card) using shadcn/ui in components/ui/

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Secure Login (Priority: P1) 🎯 MVP

**Goal**: Allow users to log in and be redirected to their role-specific workspace.

**Independent Test**: Manual verification: Log in with valid credentials and verify redirection to /admin (for admins) or /dashboard (for lawyers).

### Implementation for User Story 1

- [x] T012 [P] [US1] Create login form schema and types with Zod in components/auth/login-form.tsx
- [x] T013 [US1] Implement Login Form UI with shadcn/ui in components/auth/login-form.tsx
- [x] T014 [US1] Create login page at app/(auth)/login/page.tsx
- [x] T015 [US1] Implement 'signIn' and 'signOut' Server Actions in app/auth/actions.ts
- [x] T016 [US1] Implement auth callback route in app/auth/callback/route.ts
- [x] T017 [US1] Implement role-based redirection logic in middleware.ts after successful login
- [x] T018 [US1] Create placeholder landing pages at app/(admin)/admin/page.tsx and app/(lawyer)/dashboard/page.tsx

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - Role-Based Access Control (Priority: P2)

**Goal**: Restrict access to /admin to only admin users and redirect others.

**Independent Test**: Manual verification: Attempt to access /admin as a lawyer and verify redirection to /dashboard.

### Implementation for User Story 2

- [x] T019 [US2] Enhance middleware.ts to enforce role-based route protection for /admin and /dashboard
- [x] T020 [US2] Implement "Access Denied / Setup Required" page at app/access-denied/page.tsx
- [x] T021 [US2] Update middleware.ts to redirect users with missing profiles to /access-denied

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Final refinements and verification

- [x] T022 [P] Update README.md with project setup instructions
- [x] T023 Final manual verification of all acceptance scenarios in spec.md
- [x] T024 [P] Verify responsive design on all auth pages and dashboards
- [x] T025 Run quickstart.md validation steps

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion
- **User Stories (Phase 3+)**: All depend on Phase 2 completion
- **Polish (Final Phase)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Foundation ready - No dependencies on other stories
- **User Story 2 (P2)**: Depends on US1 for basic auth and profile structure

### Parallel Opportunities

- T002, T004, T005 in Setup can run in parallel
- T008, T009, T011 in Foundational can run in parallel
- T012 can start in parallel with UI work in US1
- T022, T024 in Polish can run in parallel

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently

### Incremental Delivery

1. Complete Setup + Foundational -> Foundation ready
2. Add User Story 1 -> Test independently -> Deploy/Demo (MVP!)
3. Add User Story 2 -> Test independently -> Deploy/Demo

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- NO automated tests allowed as per Principle III
- Each user story is independently completable and testable
