# Data Model: Intelligence Hub

## Entities

### BriefingTemplate
Stateless configuration mapping document types to briefing sections.

| Field | Type | Description |
|-------|------|-------------|
| document_type | string | enum: 'Contract', 'Evidence', 'Pleading', 'Correspondence' |
| sections | array | Ordered list of section titles and prompt instructions |

### DocumentMetadata (Enhanced)
Existing `documents` table metadata extension.

| Field | Type | Description |
|-------|------|-------------|
| is_vendor | boolean | Manually toggled at upload; used for pgvector filtering |

## Relationships
- **BriefingTemplate** maps 1:1 with **document_type** defined in Constitution Principle VII.
- **is_vendor** is a property of a **Document** which belongs to a **Client**.

## State Transitions
- **Chat**: Volatile. Message history held in React state/Vercel AI SDK and lost on refresh.
- **Briefing**: strictly on-demand (no database caching). Generated results are stored in local component state for the duration of the view.
