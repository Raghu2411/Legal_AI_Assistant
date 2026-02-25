# Research: Auth & RBAC Setup

## Decision: [Supabase SSR for Next.js 14]
- **Rationale**: Uses `@supabase/ssr` package (replaces auth-helpers) for better compatibility with App Router server/client components.
- **Alternatives considered**: `@supabase/auth-helpers-nextjs` (deprecated), standard `@supabase/supabase-js` (manual cookie management).

## Decision: [SQL Trigger for Profile Creation]
- **Rationale**: Automates `profiles` record creation on user signup, ensuring data consistency and reducing application-layer code.
- **Alternatives considered**: Manual profile creation in Server Action (prone to failure/inconsistency).

## Decision: [Role-Based Middleware Redirection]
- **Rationale**: Intercepts requests at the edge to ensure users are authenticated and redirected before page rendering.
- **Alternatives considered**: Client-side redirection (flashes content, less secure), Server Component redirection (requires boilerplate on every page).

## Decision: [shadcn/ui + React Hook Form + Zod]
- **Rationale**: Industry standard for accessible, type-safe forms with high-quality UI components.
- **Alternatives considered**: Native HTML forms (poor UX), Formik (less performant in Next.js).

## Best Practices
- **Security**: Store roles in a dedicated `profiles` table to prevent users from modifying their own roles via `app_metadata` unless strictly controlled by server-side logic.
- **Performance**: Use Middleware for session refresh and basic route protection to minimize server-side render delays.
- **Maintainability**: Centralize Supabase client initialization for client, server, and middleware to avoid initialization errors.
