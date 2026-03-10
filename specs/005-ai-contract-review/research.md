# Research: AI Contract Review (Step 5)

## Decision: Structured AI Review Prompt for Groq (Llama-3.3-70b)
**Decision**: Use a system prompt that mandates a strict JSON schema and provides a "Golden Rules" context block prioritized over the "Playbook."
**Rationale**: Llama-3.3-70b is highly performant at JSON generation. By using "Structured Output" (if supported) or strict schema enforcement, we ensure consistent parsing into the `ClauseAnalysis` entity.
**Alternatives considered**: Multiple passes for risk vs. rewrite (rejected for latency).

## Decision: TipTap for Review Studio Editor
**Decision**: Integrate TipTap with a custom extension for "Redline" marks.
**Rationale**: TipTap (Prosemirror-based) allows for high-precision text replacement via "Nodes" or "Marks," which is essential for the side-by-side 'Accept & Replace' functionality.
**Alternatives considered**: Quill (rejected due to weaker surgical replacement control), Slate (rejected as overkill).

## Decision: Server-Side Gap Analysis
**Decision**: The AI review prompt will include a "Checklist" task derived from mandatory clauses in the Golden Rules and Playbook.
**Rationale**: The LLM is best positioned to identify missing text conceptually rather than performing a simple keyword search.
**Alternatives considered**: Client-side logic for gaps (rejected as AI context is more reliable).

## Decision: PDF/DOCX Export Logic
**Decision**: Use `react-pdf` for PDF generation and `docx` (npm library) for Word export.
**Rationale**: Both libraries allow for programmatic document generation from the refined TipTap state, ensuring the final reviewed version is preserved in the desired format.
**Alternatives considered**: Server-side Puppeteer (rejected for performance and dependency weight).

## Decision: Concurrent Edit Warning
**Decision**: Simple presence detection via a `presence` table or Supabase Realtime (if enabled).
**Rationale**: As an MVP, a simple "Another lawyer is currently viewing this document" alert fulfills the requirement without needing full CRDT sync.
**Alternatives considered**: Full Document Locking (rejected as it's too restrictive for collaborative teams).
