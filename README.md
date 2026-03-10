# SAI-Legal Assistant

A highly specialized legal assistant built with Next.js, Supabase, and Groq (Llama 3.3).

## Progress Overview

- **Step 1: Auth & RBAC Setup** ✅
- **Step 2: Admin CRUD Console** ✅
- **Step 3: Client & Case Management** ✅
- **Step 4: RAG Infrastructure** ✅
- **Step 5: AI Contract Review (Review Studio)** 🚀 (In Progress)

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
- **Editor**: TipTap (ProseMirror-based) for Side-by-Side Redlining
- **Libraries**: `langchain` (@langchain/textsplitters), `pdf-parse`, `mammoth` (DOCX), `react-pdf`, `docx`

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

   ### Core Infrastructure (Steps 1-4)
   *(See `supabase/migrations/` or previous version of README for combined Core SQL)*

   ### Step 5: AI Contract Review (Review Studio)
   Run this SQL to enable the "Review Studio" features:

   ```sql
   -- 1. Enums for AI Analysis
   CREATE TYPE risk_level AS ENUM ('green', 'yellow', 'red');
   CREATE TYPE scan_status AS ENUM ('pending', 'completed', 'failed');

   -- 2. Golden Rules (Admin Control)
   CREATE TABLE golden_rules (
     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
     admin_id uuid REFERENCES profiles(id) NOT NULL,
     rule_text text NOT NULL,
     priority integer DEFAULT 0,
     is_active boolean DEFAULT true,
     created_at timestamptz DEFAULT now()
   );

   -- 3. Risk Analyses (Document-level scans)
   CREATE TABLE risk_analyses (
     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
     document_id uuid REFERENCES documents(id) ON DELETE CASCADE NOT NULL,
     timestamp timestamptz DEFAULT now(),
     version integer DEFAULT 1,
     status scan_status DEFAULT 'pending',
     raw_json jsonb DEFAULT '{}'::jsonb
   );

   -- 4. Clause Analyses (Individual findings)
   CREATE TABLE clause_analyses (
     id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
     risk_analysis_id uuid REFERENCES risk_analyses(id) ON DELETE CASCADE NOT NULL,
     original_text text NOT NULL,
     risk_status risk_level NOT NULL,
     rationale text,
     suggested_rewrite text,
     user_overridden_status risk_level,
     user_override_rationale text,
     is_gap boolean DEFAULT false,
     created_at timestamptz DEFAULT now()
   );

   -- 5. Trigger: Auto-increment Scan Version per Document
   CREATE OR REPLACE FUNCTION increment_scan_version()
   RETURNS TRIGGER AS $$
   BEGIN
     SELECT COALESCE(MAX(version), 0) + 1 
     INTO NEW.version 
     FROM risk_analyses 
     WHERE document_id = NEW.document_id;
     RETURN NEW;
   END;
   $$ LANGUAGE plpgsql;

   CREATE TRIGGER trigger_increment_scan_version
   BEFORE INSERT ON risk_analyses
   FOR EACH ROW
   EXECUTE FUNCTION increment_scan_version();

   -- 6. Row Level Security (RLS)
   ALTER TABLE golden_rules ENABLE ROW LEVEL SECURITY;
   CREATE POLICY "Anyone authenticated can view active golden rules" 
     ON golden_rules FOR SELECT 
     USING (is_active = true OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
   CREATE POLICY "Admins can manage golden rules" 
     ON golden_rules FOR ALL 
     USING ((SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

   ALTER TABLE risk_analyses ENABLE ROW LEVEL SECURITY;
   CREATE POLICY "Lawyers view own document scans" 
     ON risk_analyses FOR SELECT 
     USING (document_id IN (SELECT id FROM documents WHERE client_id IN (SELECT id FROM clients WHERE lawyer_id = auth.uid())) OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');
   CREATE POLICY "Lawyers create scans for own docs" 
     ON risk_analyses FOR INSERT 
     WITH CHECK (document_id IN (SELECT id FROM documents WHERE client_id IN (SELECT id FROM clients WHERE lawyer_id = auth.uid())) OR (SELECT role FROM profiles WHERE id = auth.uid()) = 'admin');

   ALTER TABLE clause_analyses ENABLE ROW LEVEL SECURITY;
   CREATE POLICY "Lawyers manage clause assessments" 
     ON clause_analyses FOR ALL 
     USING (risk_analysis_id IN (SELECT id FROM risk_analyses WHERE document_id IN (SELECT id FROM documents WHERE client_id IN (SELECT id FROM clients WHERE lawyer_id = auth.uid()))));
   ```

4. **Storage Setup**:
   - Ensure the `client-vaults` bucket is configured for private access.
   - Files are stored as: `client-vaults/[client_id]/[document_id]/[filename]`.

5. **Step-by-Step Step 5 Setup**:
   1. **Admin Configuration**: Log in as an Admin and navigate to the **Playbook Console**. Add your firm's "Golden Rules" (e.g., "All contracts MUST have a 30-day termination for convenience clause").
   2. **Document Readiness**: Ensure a document has been uploaded for a client and its vectorization status is 'Completed' (Step 4 prerequisite).
   3. **Enter Review Studio**: As a Lawyer, open the **Client Detail** page, find the document, and click the **Review Studio** icon.
   4. **Automated Scan**: The system will automatically trigger a Groq-powered scan using Llama 3.3.
   5. **Interactive Review**: Use the middle pane to view risks, click a risk to see the side-by-side redline, and click **Accept & Replace** to update the TipTap editor state.

## Roles & Access
- **Admin**: Full access to oversight routes (Users, Logs, Playbook, Clients) and semantic oversight. Manages global "Golden Rules."
- **Lawyer**: Access to their specific dashboard, client management, and AI retrieval. Performs interactive document reviews in the Review Studio.
- **Security**: Strict client-data isolation enforced at the database (RLS) and vector level.

## To-Do / Roadmap
- [x] **Step 5: AI Contract Review**: Review Studio with side-by-side redlining and traffic-light risk assessment.
- [ ] **Step 6: Automated Compliance**: Firm-wide dashboard for tracking compliance rates across all documents.
- [ ] **Step 7: Notifications**: Email alerts for critical audit events and risk detections.
