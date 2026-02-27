CREATE OR REPLACE FUNCTION public.retrieve_context(
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
  FROM public.embeddings e
  WHERE (e.client_id = target_client_id OR e.client_id IS NULL)
    AND 1 - (e.embedding <=> query_embedding) > match_threshold
  ORDER BY e.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
