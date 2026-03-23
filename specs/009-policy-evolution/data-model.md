# Data Model: Policy Evolution Studio

## Entities

### playbooks (Existing table, updated)
- **id**: uuid (PK)
- **name**: text (e.g., "Standard NDA")
- **content**: jsonb (Structured sections/clauses)
- **version**: int (Increments on each generation)
- **status**: text (draft, published)
- **last_updated_by**: uuid (FK: users.id)
- **last_updated_at**: timestamptz

### golden_rules (Existing table, updated)
- **id**: uuid (PK)
- **rule_text**: text
- **category**: text (e.g., "Standard vs Complex", "Governing Law")
- **priority**: int (Execution order)
- **version**: int (Increments on each update)
- **last_updated_by**: uuid (FK: users.id)
- **last_updated_at**: timestamptz

### compliance_standards (New table)
- **id**: uuid (PK)
- **name**: text
- **storage_path**: text (Supabase Storage path)
- **uploaded_by**: uuid (FK: users.id)
- **uploaded_at**: timestamptz

### policy_suggestions (New table - Transient)
- **id**: uuid (PK)
- **standard_id**: uuid (FK: compliance_standards.id)
- **target_type**: text (playbook, golden_rule)
- **target_id**: uuid (FK)
- **current_text**: text
- **suggested_text**: text
- **rationale**: text
- **status**: text (pending, approved, rejected)
- **created_at**: timestamptz

### version_history (New table - Immutable)
- **id**: uuid (PK)
- **entity_type**: text (playbook, golden_rule)
- **entity_id**: uuid
- **field**: text
- **old_value**: jsonb/text
- **new_value**: jsonb/text
- **change_type**: text (update, rollback)
- **user_id**: uuid (FK: users.id)
- **timestamp**: timestamptz

## Relationships
- A `compliance_standard` can generate many `policy_suggestions`.
- Each `playbook` update creates an entry in `version_history`.
- Each `golden_rule` update creates an entry in `version_history`.

## State Transitions
- **PolicySuggestion**: `pending` → `approved` | `rejected`.
- **Playbook**: `draft` → `published` (triggers generation and RAG sync).
