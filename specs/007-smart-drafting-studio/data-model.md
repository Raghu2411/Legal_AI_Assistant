# Data Model: Smart Drafting Studio

## New Fields (Table: `public.documents`)
| Field | Type | Description |
|-------|------|-------------|
| is_draft | boolean | True if document was created via Smart Drafting Studio |
| draft_metadata | jsonb | Stores {docType, precedents: [id1, id2], session_id} |

## New Table: `public.activity_logs`
Used to track audit trails for high-value actions (Drafting, Emails).

| Field | Type | Description |
|-------|------|-------------|
| id | uuid | Primary Key |
| user_id | uuid | Lawyer/Admin who performed the action |
| client_id | uuid | Client context |
| action_type | text | 'DRAFTING_START', 'DRAFTING_FINALIZE', 'EMAIL_GENERATED' |
| metadata | jsonb | {document_name, document_type, session_duration, email_recipient} |
| created_at | timestamptz | ISO Timestamp |

## RLS Policies
- `activity_logs`: Select/Insert allowed only for the `user_id` owner or `admin` role.
- `documents`: Existing RLS policies cover access via `client_id` (Lawyer ownership).
