# Implementation Plan: Client & Case Management

**Branch**: `003-client-management` | **Date**: 2026-02-26 | **Spec**: [specs/003-client-management/spec.md](spec.md)
**Input**: Feature specification from `/specs/003-client-management/spec.md`

## Summary
Implement a secure client and case management system with automated Case IDs, lawyer ownership, and firm-wide admin oversight. The system uses Supabase RLS for data isolation and storage for document management.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router)
**Primary Dependencies**: Supabase (Auth, DB, Storage), shadcn/ui, react-hook-form, zod
**Storage**: PostgreSQL (`clients`, `documents`), Supabase Storage (`client-vaults`)
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web application
**Project Type**: Enterprise Legal Dashboard

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Clean Code & Minimal Dependencies (I)**: ✅ Uses existing project tech stack.
- **Responsive Design (II)**: ✅ UI components follow modern responsive patterns.
- **No Testing (III)**: ✅ No automated tests planned. Manual verification per `quickstart.md`.
- **Data Isolation via RLS (IV)**: ✅ Mandatory RLS policies defined for all tables.
- **Standardized Case ID (V)**: ✅ `lawyerName-XXXX` format via database trigger.
- **Client Ownership (VI)**: ✅ Ownership and Oversight roles enforced via RLS.
- **Mandatory Document Categorization (VII)**: ✅ Schema and UI enforce Document Type selection.
- **Restricted File Formats (VIII)**: ✅ PDF, DOCX, TXT restricted via server-side validation and storage config.
- **Audit Integrity (IX)**: ✅ Reassignment already handled in Step 2; new entities follow existing audit patterns.

## Project Structure

### Documentation (this feature)

```text
specs/003-client-management/
├── spec.md              # Initial specification
├── plan.md              # This file
├── research.md          # Research on RLS and Case ID logic
├── data-model.md        # SQL schema and trigger logic
├── quickstart.md        # Manual verification guide
└── contracts/
    ├── ui-routes.md      # Application navigation structure
    └── server-actions.md # Data operation definitions
```

### Source Code (repository root)

```text
app/
├── (lawyer)/
│   └── clients/
│       ├── page.tsx       # Lawyer Client List
│       ├── new/
│       │   └── page.tsx   # Client Onboarding Form
│       └── [id]/
│           ├── page.tsx   # Client Detail View
│           └── vault/
│               └── page.tsx # Document Vault
└── (admin)/
    └── admin/
        └── clients/
            └── page.tsx   # Firm-Wide Client Management

components/
└── clients/
    ├── client-form.tsx    # Onboarding Form Component
    ├── client-table.tsx   # DataTable for list views
    ├── vault-view.tsx     # Document List and Upload Interface
    └── upload-form.tsx    # Categorized Document Upload Component

lib/
└── clients/
    └── actions.ts         # Server Actions for Client/Document operations
```

**Structure Decision**: Integrated into existing `(lawyer)` and `(admin)` route groups to maintain consistent role-based access.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No constitution violations requiring justification.*
