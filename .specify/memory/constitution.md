<!--
Sync Impact Report:
- Version change: [INITIAL] -> 1.0.0
- List of modified principles:
  - [PRINCIPLE_1_NAME] -> I. Clean Code & Minimal Dependencies
  - [PRINCIPLE_2_NAME] -> II. Responsive Design
  - [PRINCIPLE_3_NAME] -> III. No Testing (DEBT-BY-DESIGN)
  - [PRINCIPLE_4_NAME] -> IV. Role-Based Access Control (RBAC)
  - [PRINCIPLE_5_NAME] -> V. Authentication & Route Protection
- Added sections:
  - Data Isolation & Security
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/plan-template.md (✅ updated)
  - .specify/templates/spec-template.md (✅ updated - no changes needed as it's generic)
  - .specify/templates/tasks-template.md (✅ updated)
- Follow-up TODOs: None
-->

# SAI-Legal Step 1: Auth & RBAC Constitution

## Core Principles

### I. Clean Code & Minimal Dependencies
Code must be declarative and maintainable. Minimize third-party dependencies to reduce attack surface and maintenance overhead. Avoid "magic" abstractions that obscure logic.

### II. Responsive Design
The user interface must be fully responsive and functional across all device sizes (mobile, tablet, desktop). Use modern CSS layouts (Flexbox/Grid) and avoid fixed-width elements.

### III. No Testing (DEBT-BY-DESIGN)
Absolutely no unit, integration, or E2E tests are allowed for this phase. Manual verification is the only quality gate. This is a deliberate choice to accelerate initial prototyping, accepting technical debt for speed.

### IV. Role-Based Access Control (RBAC)
The system must distinguish between 'admin' and 'lawyer' roles using Supabase `app_metadata` or a dedicated `profiles` table. Admins have global access to all logs, clients, and lawyers. Lawyers have access only to their own client portfolio and associated documents.

### V. Authentication & Route Protection
No user can access any application route except `/login` without a valid session. Session management is handled via Supabase Auth. Route guards must be implemented at the application layer to enforce this.

## Data Isolation & Security

Data isolation must be enforced via Supabase Row-Level Security (RLS). All database tables must have RLS enabled and policies defined for roles ('admin', 'lawyer'). Under no circumstances should raw database access bypass these policies.

## Governance

This constitution governs Step 1 of the SAI-Legal project. All implementation must adhere to these principles. Any deviation must be documented as a "Violation" in the implementation plan.

Amendments to this constitution require a version bump. Semantic versioning is used:
- MAJOR: Principle removals or redefinitions.
- MINOR: New principle/section added.
- PATCH: Clarifications and wording fixes.

**Version**: 1.0.0 | **Ratified**: 2026-02-25 | **Last Amended**: 2026-02-25
