# Quickstart: RAG Infrastructure

**Status**: Complete | **Feature**: 004-rag-infrastructure

## Prerequisites
1. **Supabase CLI**: Required for local migration development.
2. **Mixedbread API Key**: Get your key from [mixedbread-ai](https://mixedbread.ai).
3. **Environment**: Ensure `.env.local` contains `MIXEDBREAD_API_KEY`.

## Setup Steps

### 1. Database Initialization
Run the migration script to enable `pgvector` and create the `embeddings` table with an HNSW index:
```bash
# Apply migration via Supabase SQL Editor or CLI
npx supabase migration apply
```

### 2. Service Logic
Implement the vectorization utility in `lib/ai/vector-service.ts`.
- **Chunking**: Use `RecursiveCharacterTextSplitter` (500/50).
- **Embedding**: Call `mixedbread-ai/mxbai-embed-large-v1`.
- **Storage**: Insert chunks with `client_id` (NULL for Playbook).

### 3. Automated Trigger
Update the document upload server action to call `processDocument` asynchronously.

```typescript
// Example call in upload action
const doc = await uploadDocument(file);
// Trigger vectorization (async)
vectorizeAction(doc.id);
```

### 4. UI Status Integration
Add a `VectorStatusBadge` component to the `client-vault` and `playbook` tables.
- **Pending**: 🟡 Pending
- **Processing**: 🔵 Vectorizing...
- **Ready**: 🟢 Ready
- **Error**: 🔴 Error

### 5. Manual Verification (No Testing Law III)
1. Upload a PDF with a specific legal clause.
2. Search for a semantic variant (e.g., "how to end the contract" if the clause mentions "termination").
3. Verify the result appears with >= 0.7 similarity.
4. Verify results from Client B are NOT shown for Client A.
