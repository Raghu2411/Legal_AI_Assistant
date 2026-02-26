# Research: Admin CRUD Console

## Decision: `shadcn/ui` DataTable for Administration
**Rationale**: `shadcn/ui` provides a robust, accessible, and highly customizable DataTable component based on TanStack Table. This aligns with the "Clean Code" principle by using industry-standard, maintainable UI components.
**Alternatives considered**: Building a custom table (higher maintenance), using `react-table` directly (more boilerplate).

## Decision: Hybrid Storage for Playbook & Golden Rules
**Rationale**: 
- **PDF/Docx (Files)**: Stored in Supabase Storage Bucket (`playbooks`) for efficient file handling and versioning.
- **Golden Rules (Text)**: Stored in a Postgres table (`playbooks`) for fast retrieval as context for the AI and easy editing via the Admin UI.
**Alternatives considered**: Storing everything in the DB (binary storage is inefficient), storing everything in files (parsing text context from files on every query is slow).

## Decision: AI Context Retrieval via `groq-sdk`
**Rationale**: Using the official `groq-sdk` ensures compatibility with Llama 3.3 and provides a clean, typed interface for completions.
**Alternatives considered**: Direct `fetch` calls (more error-prone, manually managing headers/types).

## Decision: Audit Trail via `logs` Table
**Rationale**: A dedicated table allows for structured logging of system events, which is essential for the 90-day retention requirement and filtering by Admin users.
**Alternatives considered**: External logging services (adds external dependency/cost), file-based logs (hard to query/filter in UI).

## Decision: PDF/Docx Parsing
**Rationale**: For Step 2, we will use a server-side parsing library (e.g., `pdf-parse` or similar) to extract text from the playbook file only when it is uploaded, storing the extracted text in the `playbooks` table for AI context retrieval. This avoids runtime parsing overhead.
**Alternatives considered**: Parsing on every AI query (slow), using a third-party document processing API (additional dependency).
