-- Drop the restrictive foreign key constraint on embeddings.document_id
-- to allow playbook IDs to be stored there as well.
ALTER TABLE public.embeddings
  DROP CONSTRAINT IF EXISTS embeddings_document_id_fkey;

-- Add vector_status and last_vectorized to playbooks table
ALTER TABLE public.playbooks 
ADD COLUMN IF NOT EXISTS vector_status text DEFAULT 'Pending',
ADD COLUMN IF NOT EXISTS last_vectorized timestamptz DEFAULT NULL;

-- Add a constraint to ensure vector_status is valid
ALTER TABLE public.playbooks
ADD CONSTRAINT playbooks_vector_status_check
CHECK (vector_status IN ('Pending', 'Processing', 'Ready', 'Error'));
