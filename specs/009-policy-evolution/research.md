# Research: Policy Evolution Studio

## Decision 1: AI Gap Analysis Engine
- **Decision**: Use a two-stage retrieval + LLM comparison approach.
- **Rationale**: Direct comparison of a large Compliance Standard against all Playbook rules in a single prompt would exceed context limits and reduce accuracy. Stage 1: Retrieve relevant Playbook/Golden Rule segments for each Compliance Standard chunk. Stage 2: Prompt LLM (Llama 3.3 via Groq) to identify contradictions, missing clauses, or required wording updates.
- **Alternatives considered**: 
  - Full-text diff: Rejected as it doesn't account for semantic meaning.
  - Multi-agent debate: Rejected as too complex for the current phase.

## Decision 2: Server-Side DOCX Generation
- **Decision**: Use the `python-docx` library in the FastAPI backend.
- **Rationale**: `python-docx` is the industry standard for programmatic DOCX creation in Python. It allows for precise control over styles, headers, and tables, which is critical for legal playbooks. The document will be generated server-side and uploaded directly to Supabase Storage.
- **Alternatives considered**: 
  - Pandoc: Powerful but requires system-level dependencies that might complicate deployment.
  - Client-side (docx.js): Rejected as the generation should be part of the secure, version-controlled backend pipeline.

## Decision 3: Real-Time Golden Rule Synchronization
- **Decision**: Module 2 (Triage) and Module 5 (Review) prompts will fetch `golden_rules` directly from the Supabase DB via a standard SELECT query at the start of each execution.
- **Rationale**: Ensures 0-latency updates. Given the low frequency of rule changes compared to triage/review executions, a direct DB fetch is efficient and ensures the "Golden Rule Precedence" (Constitution XIX) is maintained without complex cache invalidation logic.
- **Alternatives considered**: 
  - Edge Config: Faster but adds external dependency and latency for updates to propagate.
  - Redis: Increases infrastructure complexity.

## Decision 4: Playbook Storage & Diffing
- **Decision**: Store Playbook content as structured JSON in the database, with a "Draft" vs "Published" state.
- **Rationale**: JSON allows for granular "side-by-side" comparison of specific clauses in the Evolution Studio UI. It also simplifies the logic for the DOCX generator and the "Before vs After" audit trail.
- **Alternatives considered**: 
  - Raw DOCX storage only: Makes AI analysis and diffing extremely difficult.
  - Markdown files: Harder to enforce strict schema/structure than JSON.
