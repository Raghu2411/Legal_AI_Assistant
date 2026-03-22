# Research: Intelligence Hub

## Decisions

### 1. Llama 3.3 Prompt Chain with Vercel AI SDK
- **Decision**: Use `StreamData` from Vercel AI SDK to stream both the LLM response and the citation metadata (filenames, snippets) in a single response stream.
- **Rationale**: Provides a smooth UI experience while ensuring citations are available as soon as the text renders. Vercel AI SDK simplifies the management of chat history (Principle XXIII).
- **Alternatives Considered**: Manual Groq SDK streaming with custom event handlers (Rejected: Higher complexity, redundant state management).

### 2. Dynamic Briefing Engine
- **Decision**: Implement a stateless template engine in `lib/ai/briefing-templates.ts` that maps `document_type` to specific system prompts and section structures.
- **Rationale**: Principle XXV requires specialized structures. A mapping approach is scalable and keeps the logic out of the UI components.
- **Alternatives Considered**: Hardcoded switch statements in the React component (Rejected: Violates Clean Code Principle I).

### 3. Metadata-aware pgvector Filtering for Vendor Mode
- **Decision**: Use Supabase RPC calls that accept an `is_vendor_only` boolean to apply a `WHERE metadata->>'is_vendor' = 'true'` clause during vector search.
- **Rationale**: Efficiently isolates retrieval space (Step 6 requirement) while maintaining the 1024-dimension constraint of the Mixedbread model (Principle X).
- **Alternatives Considered**: Client-side filtering of retrieval results (Rejected: Inefficient, doesn't actually reduce the search space at the database level).

### 4. Interactive Footnote Badges
- **Decision**: Footnotes will render as `[1][2]` (Multi-badge per Spec clarification) using a shared `citation-badge.tsx` component. Clicking a badge will emit a custom event handled by the TipTap document viewer to trigger a scroll-to-text action.
- **Rationale**: Meets Principle XXIV and Spec requirement FR-009.
- **Alternatives Considered**: Hyperlinks to external document URLs (Rejected: Doesn't provide the "Intelligence Hub" integrated feel).
