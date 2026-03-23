import { retrieveContext } from "./vector-service";

/**
 * Searches for similar content in the vector database.
 * This is a wrapper around the more comprehensive retrieveContext function.
 * 
 * @param query The search query string (e.g., a text chunk from a standard)
 * @param limit The maximum number of results to return (default: 5)
 * @param clientId Optional client ID to filter results (default: null)
 * @returns Array of relevant content pieces with metadata
 */
export async function searchSimilarContent(query: string, limit: number = 5, clientId: string | null = null) {
  try {
    return await retrieveContext(query, clientId, 0.4, limit);
  } catch (error) {
    console.error("Error in searchSimilarContent:", error);
    return [];
  }
}
