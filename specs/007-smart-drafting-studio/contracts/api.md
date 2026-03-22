# API Contracts: Smart Drafting Studio

## POST `/api/drafting/chat`
Main interface for the interactive interview and real-time Tiptap updates.

**Request**:
```json
{
  "messages": [{"role": "user", "content": "Start drafting NDA"}],
  "clientId": "uuid",
  "docType": "NDA",
  "precedents": ["uuid1", "uuid2"],
  "documentName": "Acme NDA"
}
```

**Response**: `StreamingTextResponse`
- The text stream contains the AI's question to the user.
- A `StreamData` object (`data.append`) contains a JSON object for Tiptap:
  - `type`: "CLAUSE_UPDATE"
  - `content`: "The text of the clause generated based on the previous turn."
  - `section`: "Confidentiality" (optional)

## POST `/api/drafting/finalize`
Finalizes the draft and indexes it.

**Request**:
```json
{
  "content": "Full Tiptap HTML/JSON content",
  "clientId": "uuid",
  "metadata": {
    "name": "Acme NDA",
    "type": "Contract",
    "draft_info": { "docType": "NDA", "precedents": [...] }
  }
}
```

**Response**:
```json
{
  "success": true,
  "documentId": "uuid",
  "storagePath": "client-documents/uuid.pdf"
}
```

## POST `/api/drafting/email`
Generates a cover email draft.

**Request**:
```json
{
  "documentId": "uuid",
  "documentContent": "..."
}
```

**Response**:
```json
{
  "emailContent": "Dear Client, attached is the NDA we drafted..."
}
```
