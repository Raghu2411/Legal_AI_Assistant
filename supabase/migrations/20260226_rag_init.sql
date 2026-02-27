-- Enable the pgvector extension to work with embedding vectors
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the embeddings table
CREATE TABLE IF NOT EXISTS public.embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid REFERENCES public.documents(id) ON DELETE CASCADE,
    client_id uuid REFERENCES public.clients(id) ON DELETE CASCADE,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024) NOT NULL,
    created_at timestamptz DEFAULT now()
);

-- Create an HNSW index on the embedding column for cosine similarity
CREATE INDEX IF NOT EXISTS embeddings_embedding_idx ON public.embeddings USING hnsw (embedding vector_cosine_ops);

-- Create btree indexes for filtering
CREATE INDEX IF NOT EXISTS embeddings_document_id_idx ON public.embeddings (document_id);
CREATE INDEX IF NOT EXISTS embeddings_client_id_idx ON public.embeddings (client_id);
