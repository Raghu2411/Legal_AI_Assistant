<!--
Sync Impact Report:
- Version change: 1.0.0 -> 1.1.0
- List of modified principles:
  - IV. Role-Based Access Control (RBAC) -> IV. Lawyer Role Default & RBAC
  - V. Authentication & Route Protection -> V. Admin Access & Route Protection
- Added sections:
  - VI. Admin Resource Management
  - VII. Audit Integrity & Reassignment
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md (✅ updated)
  - .specify/templates/spec-template.md (✅ updated)
  - .specify/templates/tasks-template.md (✅ updated)
- Follow-up TODOs: None
-->

# SAI-Legal Step 2: Admin CRUD Console Constitution

## Core Principles

### I. Clean Code & Minimal Dependencies
Code must be declarative and maintainable. Minimize third-party dependencies to reduce attack surface and maintenance overhead. Avoid "magic" abstractions that obscure logic.

### II. Responsive Design
The user interface must be fully responsive and functional across all device sizes (mobile, tablet, desktop). Use modern CSS layouts (Flexbox/Grid) and avoid fixed-width elements.

### III. No Testing (DEBT-BY-DESIGN)
Absolutely no unit, integration, or E2E tests are allowed for this phase. Manual verification is the only quality gate. This is a deliberate choice to accelerate initial prototyping, accepting technical debt for speed.

### IV. Lawyer Role Default & RBAC
The system must distinguish between 'admin' and 'lawyer' roles. **All new sign-ups MUST default to the 'lawyer' role** to prevent unauthorized administrative access. RBAC is enforced via Supabase `app_metadata` or a dedicated `profiles` table.

### V. Admin Access & Route Protection
Only users with the 'admin' role can access `/admin` routes and view system-wide logs. Route guards MUST be implemented at the application layer and RLS at the database layer to enforce this restriction.

### VI. Admin Resource Management
The 'admin' role is the **sole manager** of the Legal Playbook (supporting PDF/Docx formats) and the 'Golden Rules' configuration (text field). 'Lawyers' may have read-only access if required, but never write access.

### VII. Audit Integrity & Reassignment
To preserve audit integrity, the deletion of a 'lawyer' account is a restricted operation that **MUST trigger a mandatory reassignment** of all their associated clients and documents to an 'admin' account. Data orphaned by deletion is prohibited.

## Data Isolation & Security

Data isolation must be enforced via Supabase Row-Level Security (RLS). All database tables must have RLS enabled and policies defined for roles ('admin', 'lawyer'). Under no circumstances should raw database access bypass these policies.

## Governance

This constitution governs Step 2 of the SAI-Legal project. All implementation must adhere to these principles. Any deviation must be documented as a "Violation" in the implementation plan.

Amendments to this constitution require a version bump. Semantic versioning is used:
- MAJOR: Backward incompatible governance/principle removals or redefinitions.
- MINOR: New principle/section added or materially expanded guidance.
- PATCH: Clarifications, wording, typo fixes, non-semantic refinements.

**Version**: 1.1.0 | **Ratified**: 2026-02-25 | **Last Amended**: 2026-02-25
