# API Contract: Evolution Studio

## Endpoints

### POST /api/evolution/audit
- **Purpose**: Upload a Compliance Standard and generate gap suggestions.
- **Request**: `multipart/form-data` { file: File, playbook_id: uuid }
- **Response**: `202 Accepted` { job_id: uuid }

### GET /api/evolution/suggestions/{job_id}
- **Purpose**: Retrieve the generated suggestions for review.
- **Response**: `200 OK` { suggestions: [PolicySuggestion] }

### POST /api/evolution/approve
- **Purpose**: Approve selected suggestions.
- **Request**: `application/json` { suggestion_ids: [uuid] }
- **Response**: `200 OK` { status: "success", updated_entities: [uuid] }

### GET /api/evolution/history
- **Purpose**: Fetch the full audit trail of policy changes.
- **Response**: `200 OK` { history: [VersionHistory] }

### POST /api/evolution/rollback
- **Purpose**: Restore a previous version from the history.
- **Request**: `application/json` { history_id: uuid }
- **Response**: `200 OK` { status: "success", new_version: int }
