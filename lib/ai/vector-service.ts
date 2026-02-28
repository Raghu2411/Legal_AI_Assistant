import { mxbai, EMBEDDING_MODEL } from './mixedbread';
import { getTextSplitter } from '../utils/text-splitter';
import { createAdminClient } from '../supabase/admin';

export async function processDocument(
  sourceId: string,
  clientId: string | null,
  text: string,
  sourceTable: 'documents' | 'playbooks' = 'documents'
) {
  const supabase = createAdminClient();

  try {
    // 1. Update status to Processing
    await supabase
      .from(sourceTable)
      .update({ vector_status: 'Processing' })
      .eq('id', sourceId);

    // 2. Clear existing embeddings for this document (Law XVI: Idempotency)
    await supabase
      .from('embeddings')
      .delete()
      .eq('document_id', sourceId);

    // 3. Chunk text
    const splitter = getTextSplitter();
    const rawChunks = await splitter.splitText(text);
    
    // Filter out empty or invalid chunks to avoid 422 errors
    const chunks = rawChunks.filter(chunk => typeof chunk === 'string' && chunk.trim().length > 0);

    if (chunks.length === 0) {
      throw new Error("No text chunks generated.");
    }

    // 4. Generate embeddings in batches (Mixedbread limit is 256)
    const BATCH_SIZE = 100;
    const allEmbeddings: any[] = [];

    for (let i = 0; i < chunks.length; i += BATCH_SIZE) {
      const batch = chunks.slice(i, i + BATCH_SIZE);
      let embeddingResponse;
      let retries = 0;
      const maxRetries = 3;
      
      console.log(`Processing batch ${i / BATCH_SIZE + 1} (${batch.length} chunks)...`);

      while (true) {
        try {
          embeddingResponse = await mxbai.embeddings.create({
            model: EMBEDDING_MODEL,
            input: batch,
            encoding_format: 'float'
          });
          break; // Success
        } catch (err: any) {
          if (err?.status === 429 && retries < maxRetries) {
            retries++;
            const delay = Math.pow(2, retries) * 1000;
            console.warn(`Rate limited. Retrying in ${delay}ms...`);
            await new Promise(resolve => setTimeout(resolve, delay));
          } else {
            throw err;
          }
        }
      }
      
      allEmbeddings.push(...embeddingResponse.data);
    }

    // 5. Store embeddings
    const records = chunks.map((chunk, i) => ({
      document_id: sourceId,
      client_id: clientId,
      content: chunk,
      metadata: { chunk_index: i },
      embedding: allEmbeddings[i].embedding,
    }));

    const { error: insertError } = await supabase
      .from('embeddings')
      .insert(records);

    if (insertError) {
      throw insertError;
    }

    // 6. Update status to Ready
    await supabase
      .from(sourceTable)
      .update({ 
        vector_status: 'Ready',
        last_vectorized: new Date().toISOString()
      })
      .eq('id', sourceId);

  } catch (error) {
    console.error("Error processing document:", error);
    // Update status to Error
    await supabase
      .from(sourceTable)
      .update({ vector_status: 'Error' })
      .eq('id', sourceId);
    throw error;
  }
}

export async function retrieveContext(
  query: string,
  clientId: string | null = null,
  matchThreshold = 0.7,
  matchCount = 5
) {
  const supabase = createAdminClient();

  // 1. Generate embedding for query
  const embeddingResponse = await mxbai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: [query],
    encoding_format: 'float'
  });

  const queryEmbedding = embeddingResponse.data[0].embedding;

  // 2. Call RPC
  const { data, error } = await supabase.rpc('retrieve_context', {
    query_embedding: queryEmbedding,
    match_threshold: matchThreshold,
    match_count: matchCount,
    target_client_id: clientId,
  });

  if (error) {
    console.error("Error retrieving context:", error);
    throw error;
  }

  return data;
}

export async function deleteDocumentVectors(documentId: string) {
  const supabase = createAdminClient();
  const { error } = await supabase
    .from('embeddings')
    .delete()
    .eq('document_id', documentId);

  if (error) {
    console.error("Error deleting vectors:", error);
    throw error;
  }
}
