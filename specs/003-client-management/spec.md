# Feature Specification: Client & Case Management

**Feature Branch**: `003-client-management`  
**Created**: 2026-02-26  
**Status**: Draft  
**Input**: User description: "Create technical specs for Step 3. Client & Case Management with automated Case IDs, Lawyer ownership, Admin oversight, Document categorization, and Secure storage."

## Clarifications

### Session 2026-02-26
- Q: What are the allowed values for client 'status'? → A: Active, Closed, Archived
- Q: Who is authorized to delete documents from the vault? → A: Document Owner (Lawyer) and Admin
- Q: What is the reporting window for "Recent Activity" on the Admin Dashboard? → A: Last 24 hours (Recent)
- Q: How should the system handle empty client lists in the UI? → A: Empty state with "Add Client" CTA
- Q: What is the data retention window for temporary storage and error recovery? → A: No temporary storage (direct processing)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Lawyer Client Onboarding (Priority: P1)

As a Lawyer, I want to add a new client to the system so that I can begin managing their legal cases and documents securely. When I add a client, the system should automatically generate a unique Case ID to ensure consistency with firm standards.

**Why this priority**: Essential MVP functionality. Clients are the core entity for all legal work.

**Independent Test**: Can be fully tested by a Lawyer user creating a client and verifying the client appears in their private list with a generated ID.

**Acceptance Scenarios**:

1. **Given** a logged-in Lawyer, **When** they submit the "Add Client" form with a name and case type, **Then** a new client record is created with status "Active", and the system generates a Case ID in the format `[lawyerName]-[XXXX]`.
2. **Given** a Lawyer has added a client, **When** they view their client list, **Then** only their assigned clients are visible.

---

### User Story 2 - Document Vault Management (Priority: P2)

As a Lawyer, I want to upload documents to a client's "Document Vault" and categorize them by type (Contract, Evidence, Correspondence, Pleading) so that I can maintain an organized and searchable case file.

**Why this priority**: Core value proposition of the digital assistant is document organization.

**Independent Test**: Can be tested by uploading a PDF/DOCX/TXT file to a client vault and verifying it appears with the correct category.

**Acceptance Scenarios**:

1. **Given** a Lawyer is viewing a client's Document Vault, **When** they upload a valid file (PDF, DOCX, TXT), **Then** the file is stored and displayed with its assigned category.
2. **Given** an upload attempt, **When** the file is not a PDF, DOCX, or TXT, **Then** the system rejects the upload with a clear error message.

---

### User Story 3 - Admin Firm-Wide Oversight (Priority: P3)

As an Admin, I want to view all clients across the entire firm and search for them by Lawyer Name or Client Name so that I can perform quality control and maintain firm-wide oversight.

**Why this priority**: Necessary for management and compliance, but doesn't block individual lawyer workflows.

**Independent Test**: Can be tested by an Admin user accessing the "Firm-Wide Clients" tab and searching for specific clients/lawyers.

**Acceptance Scenarios**:

1. **Given** a logged-in Admin, **When** they access the Admin Dashboard, **Then** they can see a searchable list of all clients in the system.
2. **Given** the Admin client list, **When** searching by a specific Lawyer's name, **Then** only clients assigned to that lawyer are displayed.

---

### Edge Cases

- **Duplicate Client Names**: How does the system handle two clients with the same name under the same lawyer? (Assumption: Allowed, as they will have unique Case IDs).
- **Lawyer Name Changes**: If a lawyer's name changes, does it affect existing Case IDs? (Decision: No, Case IDs are immutable once generated).
- **Empty Vaults**: Handling the UI state when a client has zero documents.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST automatically generate a unique `auto_case_id` on client creation using the format `[lawyerName]-[RandomAlphanumeric]`.
- **FR-002**: System MUST enforce Lawyer ownership: Lawyers can only see and manage clients where `lawyer_id` matches their own ID.
- **FR-003**: System MUST provide Admins with global CRUD access to all client and document records.
- **FR-010**: Document deletion MUST be restricted to the Document Owner (Lawyer who uploaded it) and system Admins.
- **FR-004**: System MUST require a 'Document Type' for every file upload (Contract, Evidence, Correspondence, Pleading).
- **FR-005**: System MUST restrict document uploads to PDF, DOCX, and TXT formats.
- **FR-006**: System MUST enforce data isolation at the database level using Supabase RLS policies based on user roles and ownership.
- **FR-007**: System MUST organize storage folders by `client_id` within the `client-vaults` bucket.
- **FR-008**: System MUST support server-side searching by Client Name and Lawyer Name in the Admin interface.
- **FR-009**: Client status MUST be restricted to the following lifecycle values: Active, Closed, Archived.
- **FR-011**: The Admin Dashboard "Recent Activity" metrics MUST reflect data from the last 24 hours.
- **FR-012**: When no clients are found for a Lawyer, the system MUST display an empty state with a call-to-action to "Add Client".
- **FR-013**: System MUST process all uploads directly; no intermediate temporary storage or deferred ingestion windows are permitted.

### Key Entities *(include if feature involves data)*

- **Client**: Represents a legal client. Attributes: ID, Auto Case ID, Name, Case Type, Lawyer ID (Owner), Status (Active/Closed/Archived).
- **Document**: Represents a file associated with a client. Attributes: ID, Client ID, File URL, Document Type (Category), Uploader ID, Timestamp.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Lawyers can onboard a new client and receive an automated Case ID in under 10 seconds.
- **SC-002**: Admins can locate any client in the firm using the search tool in under 2 seconds.
- **SC-003**: 100% of uploaded documents are correctly categorized and stored in the appropriate client folder.
- **SC-004**: Unauthorized access attempts by a Lawyer to another Lawyer's client data are blocked 100% of the time by RLS.

## Assumptions

- **AS-001**: The `lawyerName` used in the Case ID is derived from the `profiles` table associated with the `lawyer_id`.
- **AS-002**: Random suffix in Case ID is 4-6 characters to ensure uniqueness within a lawyer's portfolio.
