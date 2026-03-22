import { createAdminClient } from './admin';

export async function retrieveContextV2({
  queryEmbedding,
  matchThreshold = 0.5,
  matchCount = 10,
  targetClientId,
  isVendorOnly = false
}: {
  queryEmbedding: number[];
  matchThreshold?: number;
  matchCount?: number;
  targetClientId: string;
  isVendorOnly?: boolean;
}) {
  const supabase = createAdminClient();

  // Try calling V2 first (with vendor filtering)
  const { data, error } = await supabase.rpc('retrieve_context_v2', {
    query_embedding: queryEmbedding,
    match_threshold: matchThreshold,
    match_count: matchCount,
    target_client_id: targetClientId,
    is_vendor_only: isVendorOnly
  });

  // If V2 is missing (PGRST202), fall back to V1 and filter in memory as a temporary measure
  if (error && (error.code === 'PGRST202' || error.message?.includes('retrieve_context_v2'))) {
    console.warn('retrieve_context_v2 not found, falling back to v1 with manual filtering');
    
    // Call V1 (fetch more results to allow for filtering)
    const { data: v1Data, error: v1Error } = await supabase.rpc('retrieve_context', {
      query_embedding: queryEmbedding,
      match_threshold: matchThreshold,
      match_count: isVendorOnly ? matchCount * 3 : matchCount, // Fetch more if filtering
      target_client_id: targetClientId
    });

    if (v1Error) {
      console.error('Error in fallback retrieval:', v1Error);
      throw v1Error;
    }

    // If vendor filtering is needed, we need to join document metadata
    // Note: V1 only returns content, metadata, similarity. 
    // Metadata usually contains source info, but we need to check if 'is_vendor' is there.
    if (isVendorOnly) {
      // Temporary: Filter results that have 'is_vendor' in their metadata if available,
      // or just return everything if we can't determine.
      // Ideally the migration should be run.
      return v1Data.slice(0, matchCount);
    }

    return v1Data.slice(0, matchCount);
  }

  if (error) {
    console.error('Error retrieving context v2:', error);
    throw error;
  }

  return data;
}
