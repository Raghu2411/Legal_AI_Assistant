# Server Actions: Client & Case Management

## Client Operations
- **`createClient(formData: ClientSchema)`**:
  - Validates `name`, `case_type`.
  - Inserts into `clients` table (Case ID generated via trigger).
  - Returns `{ success: true, id: uuid }` or `{ error: string }`.

- **`updateClient(id: uuid, data: Partial<ClientSchema>)`**:
  - Updates fields in `clients` table.
  - Used by Admins for quality control.
  - Revalidate: `/clients/[id]`, `/clients`.

- **`updateClientStatus(id: uuid, status: 'Active' | 'Closed' | 'Archived')`**:
  - Updates `status` in `clients` table.
  - Revalidate: `/clients/[id]`, `/clients`.

- **`deleteClient(id: uuid)`**:
  - Restricted to `admin` role.
  - Deletes record from `clients` (Cascade delete documents).

## Document Operations
- **`uploadDocument(formData: DocumentUploadSchema)`**:
  - Validates `file` (type, existence), `doc_type`, `client_id`.
  - Uploads to `client-vaults` storage bucket.
  - Inserts record into `documents` table.
  - Returns `{ success: true }`.

- **`deleteDocument(id: uuid)`**:
  - Validates ownership or `admin` role.
  - Deletes from `client-vaults` storage.
  - Deletes from `documents` table.

## Admin Oversight
- **`getFirmClients(query: string, lawyerQuery: string)`**:
  - Fetches clients across the firm with optional search filters.
  - Joins `profiles` for lawyer name.
  - Returns `Client[]`.
