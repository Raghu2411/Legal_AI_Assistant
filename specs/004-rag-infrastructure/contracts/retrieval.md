# Retrieval Contract: RAG Infrastructure

**Interface**: `retrieve_context` (SQL Function / RPC)

## Description
Performs a semantic search for relevant text chunks, filtering by namespace isolation (client_id OR NULL) and a similarity threshold.

## Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query_embedding` | `vector(1024)` | Yes | The vector generated from the search query. |
| `target_client_id` | `uuid` | No | The specific client ID for the search context. |
| `match_threshold` | `float` | Yes | Minimum similarity score (0.7 required by Law IX). |
| `match_count` | `int` | Yes | Maximum results (Default: 5). |

## Logic / Filter (Law XI)
```sql
SELECT
  content,
  metadata,
  1 - (embedding <=> query_embedding) AS similarity
FROM embeddings
WHERE (client_id = target_client_id OR client_id IS NULL)
  AND 1 - (embedding <=> query_embedding) > match_threshold
ORDER BY similarity DESC
LIMIT match_count;
```

## Response Schema (JSON array)
```json
[
  {
    "content": "Text chunk content...",
    "metadata": { "page": 1, "doc_id": "uuid" },
    "similarity": 0.82
  }
]
```
