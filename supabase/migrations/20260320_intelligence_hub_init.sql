-- Add is_vendor boolean field to documents table
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS is_vendor boolean DEFAULT false;

-- Create pgvector retrieval RPC with metadata filtering (v2)
CREATE OR REPLACE FUNCTION public.retrieve_context_v2(
  query_embedding vector(1024),
  match_threshold float,
  match_count int,
  target_client_id uuid DEFAULT NULL,
  is_vendor_only boolean DEFAULT false
)
RETURNS TABLE (
  id uuid,
  document_id uuid,
  content text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    e.id,
    e.document_id,
    e.content,
    e.metadata,
    1 - (e.embedding <=> query_embedding) AS similarity
  FROM public.embeddings e
  JOIN public.documents d ON e.document_id = d.id
  WHERE (e.client_id = target_client_id OR e.client_id IS NULL)
    AND (NOT is_vendor_only OR d.is_vendor = true)
    AND 1 - (e.embedding <=> query_embedding) > match_threshold
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
