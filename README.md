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

- **Step 3: Client & Case Management** ✅
  - **Automated Case IDs**: Unique firm-wide IDs generated as `[lawyerName]-[XXXX]`.
  - **Lawyer Ownership**: Private client lists and document vaults for individual lawyers.
  - **Admin Oversight**: Firm-wide searchable client database.
  - **Categorized Document Vault**: Support for PDF, DOCX, and TXT with mandatory categorization (Contract, Evidence, etc.).
  - **Secure Storage**: Data isolation via Supabase RLS and client-specific folder structures.

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
   - **Step 3 SQL**: Run the following in the Supabase SQL Editor:
     ```sql
     -- Tables
     CREATE TABLE clients (
       id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
       auto_case_id text UNIQUE,
       name text NOT NULL,
       case_type text NOT NULL,
       lawyer_id uuid REFERENCES profiles(id) NOT NULL,
       status text DEFAULT 'Active',
       created_at timestamptz DEFAULT now()
     );

     CREATE TABLE documents (
       id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
       client_id uuid REFERENCES clients(id) ON DELETE CASCADE NOT NULL,
       file_url text NOT NULL,
       file_name text NOT NULL,
       doc_type text NOT NULL,
       uploaded_by uuid REFERENCES profiles(id) NOT NULL,
       uploaded_at timestamptz DEFAULT now()
     );

     -- Case ID Trigger
     CREATE OR REPLACE FUNCTION generate_client_case_id()
     RETURNS TRIGGER AS $$
     DECLARE
         lawyer_name TEXT;
         name_slug TEXT;
         random_suffix TEXT;
         final_id TEXT;
         done BOOLEAN := FALSE;
     BEGIN
         SELECT full_name INTO lawyer_name FROM profiles WHERE id = NEW.lawyer_id;
         name_slug := lower(split_part(lawyer_name, ' ', 1));
         name_slug := regexp_replace(name_slug, '[^a-z0-9]', '', 'g');
         WHILE NOT done LOOP
             random_suffix := upper(substring(replace(gen_random_uuid()::text, '-', ''), 1, 4));
             final_id := name_slug || '-' || random_suffix;
             IF NOT EXISTS (SELECT 1 FROM clients WHERE auto_case_id = final_id) THEN
                 done := TRUE;
             END IF;
         END LOOP;
         NEW.auto_case_id := final_id;
         RETURN NEW;
     END;
     $$ LANGUAGE plpgsql SECURITY DEFINER;

     CREATE TRIGGER trigger_generate_case_id
     BEFORE INSERT ON clients
     FOR EACH ROW
     EXECUTE FUNCTION generate_client_case_id();

     -- RLS Policies
     ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
     CREATE POLICY "Lawyers can view their own clients" ON clients FOR SELECT USING (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
     CREATE POLICY "Lawyers can insert their own clients" ON clients FOR INSERT WITH CHECK (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
     CREATE POLICY "Only admins can delete clients" ON clients FOR DELETE USING ((SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

     ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
     CREATE POLICY "Access via client ownership" ON documents FOR SELECT USING (EXISTS (SELECT 1 FROM clients WHERE id = documents.client_id AND (lawyer_id = auth.uid() OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin')));
     CREATE POLICY "Uploader or admin can delete documents" ON documents FOR DELETE USING (uploaded_by = auth.uid() OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
     ```

4. **Storage Setup**:
   - Create a private bucket named `playbooks`.
   - **Step 3 Bucket**: Create a private bucket named `client-vaults`.
   - **Storage RLS**: Run these policies in the SQL Editor to restrict access to client folders:
     ```sql
     -- Allow users to see files in their client's folder or if they are admin
     CREATE POLICY "Client vault access" ON storage.objects FOR SELECT TO authenticated USING (
       bucket_id = 'client-vaults' AND (
         (storage.foldername(name))[1] IN (
           SELECT id::text FROM clients WHERE lawyer_id = auth.uid()
         ) OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin'
       )
     );

     -- Allow uploads to client folder
     CREATE POLICY "Client vault upload" ON storage.objects FOR INSERT TO authenticated WITH CHECK (
       bucket_id = 'client-vaults' AND (
         (storage.foldername(name))[1] IN (
           SELECT id::text FROM clients WHERE lawyer_id = auth.uid()
         ) OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin'
       )
     );
     ```

5. **Run the development server**:
   ```bash
   npm run dev
   ```

## Roles & Access
- **Admin**: Full access to `/admin` routes (Users, Logs, Playbook, Clients) and the Lawyer dashboard.
- **Lawyer**: Access to their specific dashboard, client management, and AI assistant. Automatically defaulted on signup.
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
- [ ] **Step 4: Real-time AI Chat**: Full UI for interacting with the context-aware Llama 3.3 model.
- [ ] **Step 5: Vector Search (RAG)**: Moving from text-parsing to semantic search for larger playbooks.
- [ ] **Notifications**: Email alerts for critical audit events.
