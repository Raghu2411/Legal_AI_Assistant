-- Create obligations table
CREATE TABLE IF NOT EXISTS public.obligations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    client_id uuid NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    description text NOT NULL,
    due_date timestamptz,
    status text NOT NULL DEFAULT 'pending',
    complexity_score int CHECK (complexity_score >= 1 AND complexity_score <= 10),
    classification text CHECK (classification IN ('standard', 'complex')),
    metadata jsonb DEFAULT '{}'::jsonb,
    created_by uuid REFERENCES auth.users(id),
    created_at timestamptz DEFAULT now(),
    confirmed_at timestamptz,
    CONSTRAINT obligations_status_check CHECK (status IN ('pending', 'confirmed', 'rejected'))
);

-- Add triage columns to documents table
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS classification text DEFAULT 'standard',
ADD COLUMN IF NOT EXISTS complexity_score int DEFAULT 0,
ADD COLUMN IF NOT EXISTS triage_reasoning text,
ADD COLUMN IF NOT EXISTS triage_metadata jsonb DEFAULT '{}'::jsonb;

-- Add a constraint for classification
DO $$ BEGIN
    ALTER TABLE public.documents
    ADD CONSTRAINT documents_classification_check
    CHECK (classification IN ('standard', 'complex'));
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS obligations_document_id_idx ON public.obligations (document_id);
CREATE INDEX IF NOT EXISTS obligations_client_id_idx ON public.obligations (client_id);
CREATE INDEX IF NOT EXISTS obligations_status_idx ON public.obligations (status);
CREATE INDEX IF NOT EXISTS documents_classification_idx ON public.documents (classification);

-- RLS Policies
ALTER TABLE public.obligations ENABLE ROW LEVEL SECURITY;

-- Allow 'admin' or the 'lawyer' owner of the client to view
CREATE POLICY "Lawyers can view obligations for their clients" ON public.obligations
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM public.clients
            WHERE id = obligations.client_id
            AND (lawyer_id = auth.uid() OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
        )
    );

-- Allow lawyers and admins to insert obligations for their clients
CREATE POLICY "Lawyers can insert obligations for their clients" ON public.obligations
    FOR INSERT WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.clients
            WHERE id = obligations.client_id
            AND (lawyer_id = auth.uid() OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
        )
    );

-- Only 'admin' or the 'lawyer' owner of the client can update
CREATE POLICY "Lawyers can update obligations for their clients" ON public.obligations
    FOR UPDATE USING (
        EXISTS (
            SELECT 1 FROM public.clients
            WHERE id = obligations.client_id
            AND (lawyer_id = auth.uid() OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.clients
            WHERE id = obligations.client_id
            AND (lawyer_id = auth.uid() OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
        )
    );
