# Implementation Plan: Auth & RBAC Setup

**Branch**: `001-auth-rbac-setup` | **Date**: 2026-02-25 | **Spec**: [specs/001-auth-rbac-setup/spec.md]
**Input**: Feature specification from `/specs/001-auth-rbac-setup/spec.md`

## Summary

Implement a secure authentication system for SAI-Legal using Next.js 14+ (App Router) and Supabase. The system will enforce Role-Based Access Control (RBAC) for 'admin' and 'lawyer' roles, protected by Next.js middleware. Data isolation will be maintained via Supabase RLS and a custom 'profiles' table.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router)
**Primary Dependencies**: @supabase/auth-helpers-nextjs, @supabase/supabase-js, tailwindcss, lucide-react, shadcn/ui, react-hook-form, zod
**Storage**: PostgreSQL (Supabase)
**Testing**: [NONE - Forbidden by Constitution Principle III]
**Target Platform**: Web (Vercel)
**Project Type**: web-service
**Performance Goals**: Login completion under 2s, Middleware redirect latency <100ms
**Constraints**: Must use NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY env vars
**Scale/Scope**: Initial 2 roles (admin, lawyer), scalable profile-based RBAC

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] Principle I (Clean Code): Plan uses standard Next.js 14 patterns (Server Actions, Middleware).
- [x] Principle II (Responsive): UI built with Tailwind CSS and shadcn/ui.
- [x] Principle III (No Testing): Explicitly disabled in Technical Context.
- [x] Principle IV (RBAC): Implemented via 'profiles' table and 'role' enum.
- [x] Principle V (Auth/Route): Middleware-based route protection for all routes except /login.

## Project Structure

### Documentation (this feature)

```text
specs/001-auth-rbac-setup/
â”œâ”€â”€ plan.md              # This file
â”œâ”€â”€ research.md          # Phase 0 output
â”œâ”€â”€ data-model.md        # Phase 1 output
â”œâ”€â”€ quickstart.md        # Phase 1 output
â”œâ”€â”€ contracts/           # Phase 1 output
â””â”€â”€ tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
app/
â”œâ”€â”€ (auth)/
â”‚   â””â”€â”€ login/
â”‚       â””â”€â”€ page.tsx
â”œâ”€â”€ (lawyer)/
â”‚   â””â”€â”€ dashboard/
â”‚       â””â”€â”€ page.tsx
â”œâ”€â”€ (admin)/
â”‚   â””â”€â”€ admin/
â”‚       â””â”€â”€ page.tsx
â”œâ”€â”€ auth/
â”‚   â””â”€â”€ callback/
â”‚       â””â”€â”€ route.ts
â””â”€â”€ layout.tsx

components/
â”œâ”€â”€ ui/                  # shadcn components
â””â”€â”€ auth/
    â””â”€â”€ login-form.tsx

lib/
â”œâ”€â”€ supabase/
â”‚   â”œâ”€â”€ client.ts
â”‚   â”œâ”€â”€ server.ts
â”‚   â””â”€â”€ middleware.ts
â””â”€â”€ utils.ts

middleware.ts
```

**Structure Decision**: Option 2: Web application (Next.js App Router with route groups for role-based isolation).

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
