# SAI-Legal Assistant

A highly specialized legal assistant built with Next.js, Supabase, and Groq (Llama 3.3).

## Progress Overview

- **Step 1: Auth & RBAC Setup** ✅
  - Secure authentication with Supabase.
  - Role-Based Access Control (Admin vs. Lawyer).
  - Protected routes and automated redirections.

- **Step 2: Admin CRUD Console** ✅
  - **Admin Dashboard**: Centralized oversight with summary statistics.
  - **User Oversight**: DataTable for managing user roles and data reassignment on deletion.
  - **Audit Trail**: Detailed logging of all system activities (Login, Role Updates, Playbook Uploads).
  - **Hybrid Playbook Storage**: Version-controlled PDF uploads (Supabase Storage) + "Golden Rules" text (Postgres).
  - **AI Context Integration**: Groq (Llama 3.3) integration with firm-wide guidelines and explicit source citations.

## Tech Stack
- **Framework**: Next.js 14+ (App Router)
- **Database & Auth**: Supabase (Postgres, Auth, Storage, RLS)
- **AI/LLM**: Groq SDK (Llama 3.3 70B)
- **Parsing**: `pdf-parse` for server-side document extraction
- **Styling**: Tailwind CSS + shadcn/ui
- **Forms**: React Hook Form + Zod

## Getting Started

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Configure environment variables** in `.env.local`:
   ```env
   NEXT_PUBLIC_SUPABASE_URL=your-project-url
   NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
   GROQ_API_KEY=your-groq-api-key
   ```

3. **Database Setup**:
   - Apply migrations found in `specs/001-auth-rbac-setup/` and `specs/002-admin-crud-console/data-model.md`.
   - Create a private bucket named `playbooks` in Supabase Storage.

4. **Run the development server**:
   ```bash
   npm run dev
   ```

## Roles & Access
- **Admin**: Full access to `/admin` routes (Users, Logs, Playbook) and the Lawyer dashboard.
- **Lawyer**: Access to their specific dashboard and AI assistant. Automatically defaulted on signup.
- **Audit Integrity**: Deleting a lawyer triggers a mandatory reassignment of their data to an admin.

## Maintenance & Infrastructure

### 90-Day Log Retention (Edge Function)
To comply with the 90-day log retention principle, a Supabase Edge Function handles automated cleanup.

1. **Install Supabase CLI**:
   ```bash
   npm install supabase --save-dev
   ```

2. **Initialize and Create the Function**:
   ```bash
   npx supabase login
   npx supabase init
   npx supabase functions new cleanup-logs
   ```

3. **Deploy the Function**:
   Copy the code from the technical plan to `supabase/functions/cleanup-logs/index.ts` and run:
   ```bash
   npx supabase functions deploy cleanup-logs
   ```

4. **Schedule the Cleanup (CRON)**:
   Run the following SQL in the Supabase SQL Editor to schedule the function nightly:
   ```sql
   select
     cron.schedule(
       'cleanup-logs-nightly',
       '0 0 * * *',
       $$
       select
         net.http_post(
           url:='https://[YOUR-PROJECT-REF].functions.supabase.co/cleanup-logs',
           headers:='{"Authorization": "Bearer [YOUR-ANON-KEY]"}'::jsonb
         ) as request_id;
       $$
     );
   ```

## To-Do / Roadmap
- [ ] **Step 3: Client & Case Management**: Implementation of Case entities and document linking.
- [ ] **Step 4: Real-time AI Chat**: Full UI for interacting with the context-aware Llama 3.3 model.
- [ ] **Step 5: Vector Search (RAG)**: Moving from text-parsing to semantic search for larger playbooks.
- [ ] **Notifications**: Email alerts for critical audit events.
