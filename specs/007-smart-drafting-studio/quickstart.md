# Quickstart: Smart Drafting Studio

## Setup Requirements
1. **Groq API Key**: Ensure `GROQ_API_KEY` is set in `.env.local`.
2. **Supabase Storage**: Create the `client-documents` bucket in Supabase if not present.
3. **Database**: Run the migration to add `is_draft` and `draft_metadata` to `documents`, and create the `activity_logs` table.

## Manual Verification Flow
1. **Client Selection**:
   - Navigate to `/dashboard/drafting`.
   - Select a client from the dropdown.
   - Select a Document Type (e.g., NDA).
   - Enter a document name.
   - Click "Start Drafting".
2. **Interactive Drafting**:
   - Type a response to the AI co-pilot's first question in the Left Panel.
   - Verify that a corresponding clause appears automatically in the Tiptap editor (Right Panel).
   - Verify that the AI co-pilot asks exactly ONE follow-up question.
3. **Manual Sovereignty**:
   - Manually edit the AI-generated text in the Tiptap editor.
   - Ensure your manual changes are preserved through subsequent AI interactions.
4. **Finalization & RAG**:
   - Click "Finalize & Save".
   - Confirm the document appears in the Client's Vault.
   - Navigate to the Intelligence Hub and search for a keyword from the newly drafted document to verify RAG indexing.
5. **Email Utility**:
   - Click "Draft Email" on the saved document.
   - Verify the generated cover letter accurately reflects the document context.
