-- Create storage buckets for the Legal AI Assistant

-- 1. client-vaults (Legal documents for clients)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('client-vaults', 'client-vaults', false)
ON CONFLICT (id) DO NOTHING;

-- 2. client-documents (Finalized drafting outputs)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('client-documents', 'client-documents', false)
ON CONFLICT (id) DO NOTHING;

-- 3. compliance-standards (Admin-uploaded standards for Gap Analysis)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('compliance-standards', 'compliance-standards', false)
ON CONFLICT (id) DO NOTHING;

-- 4. firm-playbooks (Generated DOCX artifacts)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('firm-playbooks', 'firm-playbooks', false)
ON CONFLICT (id) DO NOTHING;

-- Enable RLS on storage.objects (It is enabled by default in Supabase)
ALTER TABLE storage.objects ENABLE ROW LEVEL SECURITY;

-- RLS Policies for compliance-standards (Admin only)
CREATE POLICY "Admins can manage compliance-standards"
ON storage.objects FOR ALL
TO authenticated
USING (
  bucket_id = 'compliance-standards' AND 
  (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin'
)
WITH CHECK (
  bucket_id = 'compliance-standards' AND 
  (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin'
);

-- RLS Policies for client-vaults
-- Lawyers can manage files in their own clients' folders
CREATE POLICY "Lawyers can manage their own client-vaults"
ON storage.objects FOR ALL
TO authenticated
USING (
  bucket_id = 'client-vaults' AND (
    (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin' OR
    EXISTS (
      SELECT 1 FROM public.clients 
      WHERE id::text = (storage.foldername(name))[1] 
      AND lawyer_id = auth.uid()
    )
  )
)
WITH CHECK (
  bucket_id = 'client-vaults' AND (
    (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin' OR
    EXISTS (
      SELECT 1 FROM public.clients 
      WHERE id::text = (storage.foldername(name))[1] 
      AND lawyer_id = auth.uid()
    )
  )
);

-- RLS Policies for client-documents
CREATE POLICY "Users can manage client-documents"
ON storage.objects FOR ALL
TO authenticated
USING (
  bucket_id = 'client-documents' AND (
    (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin' OR
    (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'lawyer'
  )
)
WITH CHECK (
  bucket_id = 'client-documents' AND (
    (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin' OR
    (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'lawyer'
  )
);

-- RLS Policies for firm-playbooks
CREATE POLICY "Admins can manage firm-playbooks"
ON storage.objects FOR ALL
TO authenticated
USING (
  bucket_id = 'firm-playbooks' AND 
  (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin'
)
WITH CHECK (
  bucket_id = 'firm-playbooks' AND 
  (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin'
);

CREATE POLICY "Lawyers can view firm-playbooks"
ON storage.objects FOR SELECT
TO authenticated
USING (
  bucket_id = 'firm-playbooks' AND 
  (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'lawyer'
);
