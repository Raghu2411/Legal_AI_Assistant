# Research: Smart Drafting Studio

## Groq Interview Orchestrator
**Decision**: Use a stateful system prompt with Groq (Llama 3.3 70B) to act as a legal drafting co-pilot.
**Rationale**: Groq's low latency (Llama 3.3) is ideal for real-time interactive interviews. The Vercel AI SDK `chatStream` pattern will be adapted for drafting.
**Implementation**:
- The system prompt will include the "Drafting Context" (Client info, Document Type, Precedents).
- The AI will be instructed to ask exactly one question at a time.
- The AI will output structured JSON or specific delimiters (e.g., `[[CLAUSE: ...]]`) to signal document fragments for Tiptap.

## Tiptap Integration & Real-time Updates
**Decision**: Use Tiptap's `Editor` commands to insert or update text fragments based on AI signals.
**Rationale**: Tiptap provides a clean API for programmatic document manipulation while allowing manual user input (Constitution Principle XXVIII).
**Implementation**:
- `editor.commands.insertContent()` will be used to append or replace sections.
- Placeholders `[MISSING_TERM]` will be highlighted using a custom Tiptap extension or standard marks.

## Document Storage & RAG Indexing
**Decision**: Use `lib/ai/vector-service.ts` for automatic indexing.
**Rationale**: Principle XXIX requires immediate RAG indexing upon save. `processDocument` is already implemented and proven for other features.
**Implementation**:
- Finalized documents (as PDF or DOCX) will be uploaded to the `client-documents` Supabase bucket.
- A database record in `documents` will be created with `vector_status: 'Pending'`.
- The `processDocument` function will be called as a background process to generate embeddings and update status to `Ready`.

## Email Utility & Logs
**Decision**: Create a dedicated `DraftEmailModal` that uses the finalized document content as context for a Groq-generated professional cover letter.
**Rationale**: Principle XXX requires explicit user action. 
**Implementation**:
- A 'Draft Email' button will trigger a modal.
- The email content will be generated using a specialized prompt: "Draft a professional cover email for the attached document...".
- The generated email and a log of the drafting session will be stored in a new `activity_logs` table.
