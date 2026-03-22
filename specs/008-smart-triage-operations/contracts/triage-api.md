# API Contract: Document Triage & Extraction

**Endpoint**: `POST /api/triage/process`

## Request Schema
```json
{
  "document_id": "uuid",
  "client_id": "uuid",
  "golden_rules": [
    { "keyword": "liability", "threshold": 7 }
  ]
}
```

## Response Schema (200 OK)
```json
{
  "status": "success",
  "triage": {
    "score": 8,
    "classification": "complex",
    "rationale": "High liability clause detected."
  },
  "obligations": [
    {
      "description": "Filing of quarterly reports",
      "due_date": "2026-06-30T23:59:59Z",
      "compliance": {
        "admin": "passed",
        "regulatory": "passed"
      }
    }
  ]
}
```
