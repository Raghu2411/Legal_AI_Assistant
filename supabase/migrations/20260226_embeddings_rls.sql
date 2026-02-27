-- Enable RLS on embeddings table
ALTER TABLE public.embeddings ENABLE ROW LEVEL SECURITY;

-- Policy: Users can only read embeddings for clients assigned to them
-- Or global embeddings where client_id IS NULL (Playbooks)
CREATE POLICY "Users can read assigned client embeddings"
ON public.embeddings
FOR SELECT
TO authenticated
USING (
  client_id IS NULL
  OR client_id IN (
    SELECT id FROM public.clients WHERE lawyer_id = auth.uid()
  )
);
