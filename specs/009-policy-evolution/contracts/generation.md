# API Contract: Document Generation

## Endpoints

### POST /api/generation/refresh-playbook
- **Purpose**: Manually trigger a fresh generation and indexing for a playbook.
- **Request**: `application/json` { playbook_id: uuid, format: "docx" | "pdf" }
- **Response**: `200 OK` { status: "success", version: int, storage_url: text }

### GET /api/generation/playbook-status/{playbook_id}
- **Purpose**: Check the generation and RAG sync status.
- **Response**: `200 OK` { status: "pending" | "success" | "failed", storage_url: text, last_indexed_at: timestamptz }
