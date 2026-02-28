# Implementation Plan: Admin CRUD Console

**Branch**: `002-admin-crud-console` | **Date**: 2026-02-25 | **Spec**: [specs/002-admin-crud-console/spec.md]
**Input**: Feature specification from `/specs/002-admin-crud-console/spec.md`

## Summary

Implement the Admin CRUD Console for Step 2. This includes a centralized /admin dashboard using shadcn/ui DataTables for user oversight and audit logs. The system will use hybrid storage (Supabase Storage for playbook files and Postgres for Golden Rules) and integrate this context into the AI assistant (Llama 3.3 via Groq) with explicit source citations.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router)
**Primary Dependencies**: @supabase/supabase-js, @supabase/ssr, groq-sdk, lucide-react, shadcn/ui components (DataTable, Table, Button, etc.), pdf-parse (for playbook parsing)
**Storage**: 
- **PostgreSQL**: `profiles` (roles), `logs` (audit trail), `playbooks` (golden rules text + metadata).
- **Supabase Storage**: `playbooks` bucket (PDF/Docx versions).
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web
**Project Type**: Next.js Full-stack Web Application
**Performance Goals**: Dashboard page load < 2s; AI response context retrieval < 500ms.
**Constraints**: 
- Admin role required for /admin access.
- Lawyer role is the default for new sign-ups.
- 90-day retention for audit logs.
- Mandatory reassignment of data on lawyer deletion.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] I. Clean Code & Minimal Dependencies: Using industry-standard libraries.
- [x] II. Responsive Design: Tailwind CSS for layout.
- [x] III. No Testing: Manual verification confirmed.
- [x] IV. Lawyer Role Default & RBAC: Enforced in signup logic and route guards.
- [x] V. Admin Access & Route Protection: Enforced via Middleware and RLS.
- [x] VI. Admin Resource Management: Playbook/Rules restricted to Admin.
- [x] VII. Audit Integrity & Reassignment: Mandatory reassignment logic in lib/supabase/admin.ts.

## Project Structure

### Documentation (this feature)

```text
specs/002-admin-crud-console/
├── plan.md              # This file
├── research.md          # Research findings
├── data-model.md        # DB and Storage schema
├── quickstart.md        # Validation steps
├── contracts/           
│   └── ui-routes.md     # Admin UI structure
└── tasks.md             # Implementation tasks
```

### Source Code (repository root)

```text
app/
├── (admin)/
│   └── admin/
│       ├── layout.tsx          # Admin layout with Sidebar
│       ├── page.tsx            # Dashboard Overview
│       ├── users/
│       │   └── page.tsx        # User Oversight
│       ├── logs/
│       │   └── page.tsx        # Audit Trail
│       └── playbook/
│           └── page.tsx        # Playbook Management
├── auth/
│   └── actions.ts              # Updated signup (default role: lawyer)
lib/
├── supabase/
│   ├── client.ts
│   ├── server.ts
│   ├── middleware.ts
│   └── admin.ts                # Admin-level reassignment logic
├── ai/
│   └── groq.ts                 # AI Logic with context
components/
├── admin/
│   ├── user-table/             # User DataTable components
│   ├── log-table/              # Log DataTable components
│   └── playbook-form.tsx       # Hybrid upload form
└── ui/                         # shadcn/ui components
```

**Structure Decision**: Next.js App Router with Route Groups `(admin)` for layout isolation and `lib/` for shared utilities.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

*No violations.*
