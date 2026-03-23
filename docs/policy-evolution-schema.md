# Policy Evolution & Rollback Procedures

## Database Schema: `version_history`

| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Primary Key |
| entity_type | text | 'playbook' or 'golden_rule' |
| entity_id | uuid | ID of the target rule/playbook |
| field | text | Specific field changed (e.g., 'rule_text') |
| old_value | jsonb | State before the change |
| new_value | jsonb | State after the change |
| change_type | text | 'update', 'rollback', or 'generation' |
| user_id | uuid | Admin who performed the action |
| timestamp | timestamptz | When the change occurred |

## Rollback Procedure

When a rollback is triggered via the Evolution Studio:
1. The system fetches the `old_value` from the selected history entry.
2. It increments the `version` of the target entity.
3. It performs an atomic update to restore the `old_value`.
4. A NEW history entry is created with `change_type: 'rollback'` to maintain an immutable audit trail.
5. If the entity is a Playbook, a new RAG synchronization is automatically triggered to ensure the vector database reflects the restored state.

## Concurrency Protection

To prevent "Lost Updates" in a multi-admin environment:
- The UI tracks the `version` of the rules it displays.
- When submitting approvals, the backend validates that the current DB version matches the UI's loaded version.
- If a conflict is detected (HTTP 409), the update is rejected and the admin is prompted to refresh.
