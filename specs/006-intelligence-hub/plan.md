# Implementation Plan: Intelligence Hub

**Branch**: `006-intelligence-hub` | **Date**: 2026-03-20 | **Spec**: [specs/006-intelligence-hub/spec.md]
**Input**: Feature specification from `/specs/006-intelligence-hub/spec.md`

## Summary

The Intelligence Hub (Step 6) introduces a centralized, tabbed interface for advanced client document interaction. It features a conversational AI Chat with session memory (using Vercel AI SDK and Llama 3.3 via Groq), a Dynamic Briefing Generator that adapts summaries to document types (Contract, Evidence, etc.), and a specialized Vendor Mode for isolated procurement risk analysis. All AI responses are strictly grounded in document context with numbered, interactive citations linking back to the source material.

## Technical Context

**Language/Version**: TypeScript / Next.js 14+ (App Router)
**Primary Dependencies**: Vercel AI SDK (`ai`), Groq SDK (`groq-sdk`), shadcn/ui (Tabs, ScrollArea), Lucide React
**Storage**: Supabase (Postgres for Metadata, pgvector for RAG), volatile in-memory chat state
**Testing**: NONE - Forbidden by Constitution Principle III
**Target Platform**: Web (Responsive)
**Project Type**: Web Application (Intelligence Hub Feature)
**Performance Goals**: <100ms tab switching, <5s briefing generation
**Constraints**: Zero hallucinations (Constitution Principle XXII), Strictly grounded retrieval
**Scale/Scope**: Client document vaults (varied sizes), multi-source retrieval (5-7 snippets)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle XXII (Strict Retrieval Constraint)**: AI MUST only answer from provided context. Implementation will use strict system prompting. ✅
- **Principle XXIII (Session Context Persistence)**: Chat history MUST be passed to LLM. Implementation uses Vercel AI SDK session state. ✅
- **Principle XXIV (Source Attribution)**: Footnotes MUST be interactive. UI will implement clickable badges linking to document segments. ✅
- **Principle XXV (Dynamic Briefing Templates)**: Template engine MUST adapt to document type. Implementation uses specialized prompt templates. ✅
- **Principle III (No Testing)**: Manual verification only. ✅

## Project Structure

### Documentation (this feature)

```text
specs/006-intelligence-hub/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── chat-ui.md       # UI interaction contract
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
app/
└── (lawyer)/
    └── intelligence-hub/
        ├── page.tsx          # Main Hub Tab layout
        └── actions.ts        # Server Actions for briefings/chat
components/
└── intelligence-hub/
    ├── chat-panel.tsx        # Vercel AI SDK integration
    ├── briefing-panel.tsx    # Dynamic template renderer
    ├── vendor-toggle.tsx     # pgvector filter switch
    └── citation-badge.tsx    # Interactive footnote component
lib/
├── ai/
│   ├── briefing-templates.ts # Template definitions
│   └── chat-stream.ts        # Llama 3.3 prompt chain logic
└── supabase/
    └── vector-queries.ts     # Metadata-aware pgvector filters
```

**Structure Decision**: Integrated into existing Lawyer dashboard under a new `intelligence-hub` route to maintain logical separation while reusing Supabase/AI shared libs.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
