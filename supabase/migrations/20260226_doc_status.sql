-- Add vector_status and last_vectorized columns to documents table
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS vector_status text DEFAULT 'Pending',
ADD COLUMN IF NOT EXISTS last_vectorized timestamptz DEFAULT NULL;

-- Add a constraint to ensure vector_status is valid
ALTER TABLE public.documents
ADD CONSTRAINT documents_vector_status_check
CHECK (vector_status IN ('Pending', 'Processing', 'Ready', 'Error'));
