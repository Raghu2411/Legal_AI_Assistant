# Research: RAG Infrastructure Implementation

**Status**: Complete | **Feature**: 004-rag-infrastructure

## Decision 1: Vector Storage & Indexing
**Decision**: Use Supabase (PostgreSQL) with `pgvector` and an **HNSW (Hierarchical Navigable Small World)** index on the `embedding` column.
**Rationale**: HNSW is the industry standard for high-performance vector retrieval in PostgreSQL. It is significantly faster than IVFFlat for large datasets and provides better recall at low latency.
**Alternatives considered**: IVFFlat (rejected due to re-indexing requirements and lower performance on growing datasets).

## Decision 2: Chunking Strategy
**Decision**: Use `RecursiveCharacterTextSplitter` from `langchain` with a chunk size of 500 characters and a 50-character overlap.
**Rationale**: This configuration balances retrieval granularity with semantic context. Recursive splitting ensures that logical boundaries (paragraphs, sentences) are respected as much as possible, while the overlap prevents critical information from being lost at chunk boundaries.
**Alternatives considered**: Fixed-size chunking (rejected as it breaks sentences/concepts randomly).

## Decision 3: Pipeline Trigger & Background Processing
**Decision**: Use **Next.js Server Actions** for manual/triggered vectorization with a database status field (`vectorization_status`) to manage async UI feedback.
**Rationale**: Server Actions are simpler to implement and secure in Next.js 14. Using a status field ('Pending', 'Processing', 'Ready', 'Error') allows the UI to update asynchronously without a heavy background job runner.
**Alternatives considered**: Supabase Edge Functions + Webhooks (considered but Server Actions provide better developer experience for the current Next.js-centric stack).

## Decision 4: Retrieval & Filtering (Dual-Namespace)
**Decision**: Implement a custom SQL function (RPC) or `match_embeddings` stored procedure that takes `query_embedding`, `target_client_id`, and `threshold`.
**Rationale**: Performing filtering (Client ID OR NULL) directly in the vector similarity query is the most performant and secure way to enforce RAG laws XI and XII.
**Alternatives considered**: Client-side filtering (rejected due to security and performance concerns).

## Decision 5: Atomic Batching & Failures
**Decision**: Wrap `processDocument` in a transaction or use a cleanup-on-failure pattern. If a batch fails, the document's status is set to 'Error' and any partially created vectors for that `document_id` are purged.
**Rationale**: Complies with Law XV (Failure Atomicity) and ensures no partial or inconsistent data exists in the `embeddings` table.
**Alternatives considered**: Sequential individual inserts (rejected due to inconsistency risk).
