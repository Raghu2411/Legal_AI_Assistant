# SAI-Legal Assistant

Step 1: Auth & RBAC Setup.

## Tech Stack
- Next.js 14+ (App Router)
- Supabase (Auth, SSR, RLS)
- Tailwind CSS + shadcn/ui
- React Hook Form + Zod

## Getting Started

1. Install dependencies:
   ```bash
   npm install
   ```

2. Configure environment variables in `.env.local`:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=your-project-url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
   ```

3. Initialize the database using the SQL in `specs/001-auth-rbac-setup/quickstart.md`.

4. Run the development server:
   ```bash
   npm run dev
   ```

## Roles & Access
- **Admin**: Access to `/admin` and `/dashboard`.
- **Lawyer**: Access to `/dashboard` only. Redirection from `/admin` to `/dashboard`.
- **Protected Routes**: All routes except `/login` require an active session.
