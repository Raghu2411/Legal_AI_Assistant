# SAI-Legal Assistant

A highly specialized legal assistant built with Next.js, Supabase, and Groq (Llama 3.3).

## Progress Overview

- **Step 1: Auth & RBAC Setup** ✅
- **Step 2: Admin CRUD Console** ✅
- **Step 3: Client & Case Management** ✅
- **Step 4: RAG Infrastructure** ✅

## 📂 Project Structure: SQL & Policies
All database logic is version-controlled in the following locations:
- **Core Migrations**: `supabase/migrations/`
- **Setup Scripts**: Listed below in the [Database Setup](#database-setup) section.
- **Data Model Docs**: `specs/[feature-name]/data-model.md`

---

## Tech Stack
- **Framework**: Next.js 14+ (App Router)
- **Database & Auth**: Supabase (Postgres, Auth, Storage, RLS, pgvector)
- **AI/LLM**: Groq SDK (Llama 3.3 70B) + Mixedbread AI (Embeddings)
- **Libraries**: `langchain` (@langchain/textsplitters), `pdf-parse`, `mammoth` (DOCX)

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
   MIXEDBREAD_API_KEY=your-mixedbread-api-key
   ```

<a name="database-setup"></a>
3. **Database Setup**:
   The following SQL should be run in the **Supabase SQL Editor** to initialize the system. 

   > **Note**: For local development, these files are also available in `supabase/migrations/`.

   ### Step 4 & Full Sync SQL (Tables & Logic)
   ```sql
   -- 1. Enable pgvector
   CREATE EXTENSION IF NOT EXISTS vector;

   -- 2. Core Tables
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
     uploaded_at timestamptz DEFAULT now(),
     vector_status text DEFAULT 'Pending',
     last_vectorized timestamptz
   );

   CREATE TABLE playbooks (
     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
     file_path text,
     file_name text,
     golden_rules text,
     version integer DEFAULT 1,
     created_by uuid REFERENCES profiles(id),
     created_at timestamptz DEFAULT now(),
     vector_status text DEFAULT 'Pending',
     last_vectorized timestamptz
   );

   CREATE TABLE embeddings (
     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
     document_id uuid NOT NULL, -- Reference to documents.id or playbooks.id
     client_id uuid REFERENCES clients(id), -- NULL for global firm-wide data
     content text NOT NULL,
     metadata jsonb DEFAULT '{}'::jsonb,
     embedding vector(1024) NOT NULL,
     created_at timestamptz DEFAULT now()
   );

   -- 3. Indexes & Semantic Search Logic
   CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);
   CREATE INDEX ON embeddings (document_id);
   CREATE INDEX ON embeddings (client_id);

   CREATE OR REPLACE FUNCTION retrieve_context(
     query_embedding vector(1024),
     match_threshold float,
     match_count int,
     target_client_id uuid DEFAULT NULL
   )
   RETURNS TABLE (
     content text,
     metadata jsonb,
     similarity float
   )
   LANGUAGE plpgsql
   AS $$
   BEGIN
     RETURN QUERY
     SELECT
       e.content,
       e.metadata,
       1 - (e.embedding <=> query_embedding) AS similarity
     FROM embeddings e
     WHERE (e.client_id = target_client_id OR e.client_id IS NULL)
       AND 1 - (e.embedding <=> query_embedding) > match_threshold
     ORDER BY similarity DESC
     LIMIT match_count;
   END;
   $$;

   -- 4. Case ID Generator Trigger
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
   ```

   ### Row Level Security (RLS) Policies
   ```sql
   -- Clients
   ALTER TABLE clients ENABLE ROW LEVEL SECURITY;
   CREATE POLICY "Lawyers view own or admin view all" ON clients FOR SELECT USING (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
   CREATE POLICY "Lawyers insert own" ON clients FOR INSERT WITH CHECK (auth.uid() = lawyer_id OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

   -- Embeddings
   ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;
   CREATE POLICY "Users read assigned or global" ON public.embeddings FOR SELECT TO authenticated
   USING (client_id IS NULL OR client_id IN (SELECT id FROM public.clients WHERE lawyer_id = auth.uid()) OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
   ```

4. **Storage Setup**:
   - Create private buckets: `playbooks` and `client-vaults`.
   - **Storage RLS**:
     ```sql
     CREATE POLICY "Vault access" ON storage.objects FOR SELECT TO authenticated USING (
       bucket_id = 'client-vaults' AND (
         (storage.foldername(name))[1] IN (SELECT id::text FROM clients WHERE lawyer_id = auth.uid()) 
         OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin'
       )
     );
     ```

5. **Run the development server**:
   ```bash
   npm run dev
   ```

## Roles & Access
- **Admin**: Full access to oversight routes (Users, Logs, Playbook, Clients) and semantic oversight.
- **Lawyer**: Access to their specific dashboard, client management, and AI retrieval.
- **Security**: Strict client-data isolation enforced at the vector level via dual-namespace filtering.

## To-Do / Roadmap
- [ ] **Step 5: Real-time AI Chat**: Full UI for interacting with the context-aware Llama 3.3 model.
- [ ] **Step 6: Automated Compliance**: Automated checks of client documents against the firm playbook.
- [ ] **Step 7: Notifications**: Email alerts for critical audit events.
