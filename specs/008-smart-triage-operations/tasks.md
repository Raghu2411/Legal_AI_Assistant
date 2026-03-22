# Tasks: Smart Triage & Operations

## Phase 1: Setup

- [x] T001 Create database migration for `obligations` table in `supabase/migrations/20260322_obligations_schema.sql`
- [x] T002 Apply `obligations` table migration to Supabase local instance (Skipped - Docker not running)
- [x] T003 [P] Create directory structure for Operations Dashboard in `app/(lawyer)/dashboard/operations/`
- [x] T004 [P] Create directory structure for Operations components in `components/operations/`

## Phase 2: Foundational

- [x] T005 [P] Implement base layout for Operations Dashboard in `app/(lawyer)/dashboard/operations/layout.tsx`
- [x] T006 [P] Implement shared Groq utility for triage and extraction in `lib/ai/groq-client.ts`
- [x] T007 [P] Define TypeScript interfaces for Triage and Obligations in `lib/ai/types.ts`

## Phase 3: User Story 1 - Automated Document Triage (P1)

**Story Goal**: Automatically categorize uploaded documents based on firm "Golden Rules" using AI.

- [x] T008 [P] [US1] Implement classification logic using Llama 3.3 in `lib/ai/triage-service.ts`
- [x] T009 [US1] Create API route for document triage in `app/api/triage/process/route.ts`
- [x] T010 [US1] Implement Triage Queue table component in `components/operations/triage-table.tsx` using shadcn/ui
- [x] T011 [US1] Add manual override modal with mandatory reason field in `components/operations/triage-override-modal.tsx`
- [x] T012 [US1] Integrate Triage Queue into main Operations page `app/(lawyer)/dashboard/operations/page.tsx`

## Phase 4: User Story 2 - Obligation Extraction & Verification (P2)

**Story Goal**: Extract legal obligations (dates, tasks) for manual lawyer confirmation.

- [x] T013 [P] [US2] Implement dual-scope compliance (Admin vs Regulatory) and obligation extraction logic in `lib/ai/extractor.ts` using Groq
- [x] T014 [US2] Create obligation item component with Confirm/Reject actions in `components/operations/obligation-item.tsx`
- [x] T015 [US2] Implement verification list view in `components/operations/verification-list.tsx`
- [x] T016 [US2] Add 'TBD' date handling logic in extraction service and UI
- [x] T017 [US2] Implement Server Actions for obligation confirmation, manual addition, and editing in `lib/operations/actions.ts`
- [x] T017b [US2] Implement "Manual Add Obligation" form component in `components/operations/manual-obligation-form.tsx`
- [x] T017c [US2] Implement "Edit Obligation" modal component in `components/operations/edit-obligation-modal.tsx`

## Phase 5: User Story 3 - Operational Calendar Management (P3)

**Story Goal**: Visualize confirmed obligations on a unified calendar.

- [x] T018 [P] [US3] Implement Calendar view component in `components/operations/calendar-view.tsx` using shadcn/ui
- [x] T019 [US3] Integrate Calendar view into the Operations Dashboard with tab switching
- [x] T020 [US3] Add obligation detail popover for calendar entries in `components/operations/calendar-item-popover.tsx`
- [x] T021 [US3] Implement data fetching for confirmed obligations to populate the calendar

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T022 [P] Implement dual-scope compliance flagging UI indicators in `components/operations/compliance-sidebar.tsx`
- [x] T023 Implement activity logging for all triage and confirmation events in `lib/ai/triage-service.ts` and `lib/operations/actions.ts`
- [x] T024 [P] Add loading states and error handling for AI processing in Triage Queue
- [x] T025 Final UI alignment with shadcn/ui theme and mobile responsiveness check
