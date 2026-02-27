# Data Model: RAG Infrastructure

**Status**: Complete | **Feature**: 004-rag-infrastructure

## Entities

### `embeddings` (New Table)
Represents a text chunk and its corresponding vector representation.

| Column | Type | Description |
|--------|------|-------------|
| `id` | `uuid` | Primary Key, `gen_random_uuid()` |
| `document_id` | `uuid` | Reference to `documents.id`, `ON DELETE CASCADE` |
| `client_id` | `uuid` | Reference to `clients.id`, `NULLABLE` (NULL for global Playbook) |
| `content` | `text` | The raw text of the chunk |
| `metadata` | `jsonb` | Metadata including `chunk_index`, `page_number`, etc. |
| `embedding` | `vector(1024)` | 1024-dimensional vector from Mixedbread AI |
| `created_at` | `timestamptz` | `now()` |

**Indexes**:
- `HNSW index` on `embedding` using `vector_cosine_ops`.
- `btree` index on `document_id` for fast deletion/lifecycle sync.
- `btree` index on `client_id` for namespace filtering.

### `documents` (Modified Entity)
Updates to the existing document record for status tracking.

| New Column | Type | Default | Description |
|------------|------|---------|-------------|
| `vector_status` | `text` | `'Pending'` | `'Pending'`, `'Processing'`, `'Ready'`, `'Error'` |
| `last_vectorized` | `timestamptz` | `NULL` | Timestamp of last successful vectorization |

## Relationships
- **One-to-Many**: One `document` has many `embeddings`.
- **Many-to-One**: Many `embeddings` belong to one `client` (optional).

## State Transitions (Document Vector Status)
1. **Pending**: Initial state upon file upload (FR-006).
2. **Processing**: Transitioned when `processDocument` starts.
3. **Ready**: Transitioned on successful batch embedding storage (SC-001).
4. **Error**: Transitioned if chunking, extraction, or API call fails (Law XV).

## Identity & Uniqueness
- Each chunk is uniquely identified by its `id`.
- The `embeddings` table does not enforce uniqueness on `document_id`, as multiple chunks exist per document.
- Law XVI (Idempotency) requires purging `existing embeddings` where `document_id = X` before starting a new `Ready` transition.
