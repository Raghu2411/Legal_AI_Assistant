# Data Model: Admin CRUD Console

## Database Schema (PostgreSQL)

### Table: `profiles` (Existing - Updated)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key (references auth.users) |
| email | text | User email |
| role | text | 'admin' or 'lawyer' (Default: 'lawyer') |
| created_at | timestamp | Account creation date |

### Table: `logs` (New - Audit Trail)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key |
| user_id | uuid | Foreign Key (references profiles.id) |
| event_type | text | 'LOGIN', 'ROLE_UPDATE', 'PLAYBOOK_UPLOAD', etc. |
| description | text | Detailed description of the event |
| metadata | jsonb | Additional context (IP, user agent, old/new values) |
| created_at | timestamp | Event timestamp |

### Table: `playbooks` (New - Configuration)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key |
| file_path | text | Path in Supabase storage bucket |
| file_name | text | Display name of the playbook file |
| golden_rules | text | Firm-wide principles text |
| version | integer | Incremental version number |
| created_by | uuid | Foreign Key (references profiles.id) |
| created_at | timestamp | Upload/Update timestamp |

## Storage Structure (Supabase Storage)

### Bucket: `playbooks`
- **Policy**: `authenticated` users can read. Only `admin` role can write/delete.
- **Structure**:
  - `versions/playbook_v[NUMBER].pdf` (or .docx)

## AI Context Structure

### Context Payload
```json
{
  "system_prompt": "You are a legal assistant. Use the following firm guidelines...",
  "golden_rules": "[TEXT FROM playbooks.golden_rules]",
  "playbook_content": "[EXTRACTED TEXT OR REFERENCE FROM playbooks.file_path]",
  "user_query": "[QUERY]"
}
```
*Note: Playbook content extraction may require a separate service or library depending on the complexity of parsing PDF/Docx.*
