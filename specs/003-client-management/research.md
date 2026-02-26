# Research: Client & Case Management

## Objective
Implement a robust client and document management system with automated Case IDs, Lawyer ownership, and Admin oversight, ensuring data isolation via Supabase RLS.

## Technical Decisions

### 1. Database Schema
- **Decision**: Create `clients` and `documents` tables with foreign keys to `profiles`.
- **Rationale**: Standard relational pattern for Step 3 entities. `clients` tracks ownership (`lawyer_id`), while `documents` links files to clients.
- **Alternatives Considered**: JSONB storage for documents (rejected for lack of indexing and relational integrity).

### 2. Case ID Generation
- **Decision**: Implement a PostgreSQL trigger `trigger_generate_case_id` calling a function `generate_client_case_id()`.
- **Rationale**: Ensures IDs are generated at the source of truth (DB) regardless of the client (Web, API, CLI). Follows constitution principle V.
- **Alternatives Considered**: App-side generation (rejected for potential collisions and inconsistency).

### 3. Row-Level Security (RLS)
- **Decision**: 
  - `clients`: Lawyers see/insert their own (`lawyer_id = auth.uid()`). Admins see all. Deletions restricted to Admins.
  - `documents`: Access granted via client ownership/oversight. Deletions restricted to uploader or Admin.
- **Rationale**: Aligns with Constitution Principle IV (Data Isolation via RLS) and Principle VI (Client Ownership).
- **Alternatives Considered**: Application-level filtering (rejected as per Constitution Principle IV).

### 4. Storage Structure & Policies
- **Decision**: Private bucket `client-vaults`. Folder structure: `/[client_id]/[document_id]_[filename]`.
- **Rationale**: Isolates files by client. `client_id` in path allows for efficient RLS policies on `storage.objects`.
- **Alternatives Considered**: Flat structure (rejected for difficulty in applying granular access control).

### 5. Search Implementation
- **Decision**: Server-side searching using Supabase `.ilike()` and joins.
- **Rationale**: Admin search requires joining `clients` with `profiles` (for lawyer name). Server-side ensures performance as the firm grows.
- **Alternatives Considered**: Client-side filtering (rejected for scalability issues).

### 6. File Format Enforcement
- **Decision**: Multi-layer validation (HTML `accept` attribute, Zod schema validation in Server Actions, and Supabase Storage bucket configuration).
- **Rationale**: Defense-in-depth approach to ensure only PDF, DOCX, and TXT are stored, as per Constitution Principle VIII.
- **Alternatives Considered**: Single-point validation (rejected for security/robustness).

## Unknowns Resolved
- **Case ID Format**: Confirmed as `[lawyerName]-[XXXX]` where lawyer name is the first word of `profiles.full_name`.
- **Client Status**: Lifecycle defined as `Active`, `Closed`, `Archived`.
- **Admin Edit Modal**: Confirmed as a requirement for quality control over client details.
