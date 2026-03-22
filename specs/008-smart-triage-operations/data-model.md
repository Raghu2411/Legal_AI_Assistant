# Data Model: Smart Triage & Operations

## Tables

### `obligations`
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key |
| document_id | uuid | FK to `documents(id)` |
| client_id | uuid | FK to `clients(id)` |
| description | text | Details of the task or milestone |
| due_date | timestamptz | Extracted or manually assigned date |
| status | text | 'pending', 'confirmed', 'rejected' |
| complexity_score | int | 1-10 assigned by AI |
| classification | text | 'standard', 'complex' |
| metadata | jsonb | Stores compliance flagging details |
| created_by | uuid | FK to `auth.users(id)` (Lawyer who confirmed) |
| created_at | timestamptz | Default now() |
| confirmed_at | timestamptz | Timestamp of confirmation |

## State Transitions
1. **Extraction**: Document → AI → `obligations` (status='pending')
2. **Confirmation**: Lawyer → UI → `obligations` (status='confirmed', confirmed_at=now())
3. **Rejection**: Lawyer → UI → `obligations` (status='rejected')

## RLS Policies
- `SELECT`: Only 'admin' or the 'lawyer' owner of the client can view.
- `INSERT`: Triggered by AI service (service role) or Lawyer manual entry.
- `UPDATE`: Only 'admin' or the 'lawyer' owner of the client can update status.
