# Feature Specification: Admin CRUD Console

**Feature Branch**: `002-admin-crud-console`  
**Created**: 2026-02-25  
**Status**: Draft  
**Input**: Create technical specs for Step 2. - UI: Admin Dashboard with shadcn/ui DataTable for user oversight and audit logs. - Playbook: Implement a hybrid storage system (Supabase Bucket for files + Postgres for 'Golden Rules' text). - Database: Create a 'logs' table for the Audit Trail and a 'playbooks' table for firm-wide rules. - Logic: Ensure the AI context (Llama 3.3 via Groq) will pull from both the uploaded Playbook and the Golden Rules text. - Gorq API (GROQ_API_KEY) is available.

## Clarifications

### Session 2026-02-25
- Q: Audit Log Retention → A: Retain all for 90 days.
- Q: Playbook Versioning → A: Maintain version history (Keep old files).
- Q: User Oversight Actions → A: View details and change roles ('admin' <-> 'lawyer').
- Q: AI Response Citations → A: Mention source names (e.g., "Per Golden Rules...", "According to Playbook...").
- Q: Rule Conflict Resolution → A: AI should highlight the conflict to the user.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - User Oversight & Audit Logs (Priority: P1)

As an Administrator, I want a centralized dashboard to view all registered users and a detailed history of system activities so that I can maintain oversight and ensure accountability within the firm.

**Why this priority**: Oversight of users and logs is critical for security and compliance, ensuring that only authorized personnel have access and that all actions are traceable.

**Independent Test**: Can be tested by navigating to the `/admin` dashboard as an 'admin' user and verifying that a list of users and a history of logs are correctly displayed and searchable.

**Acceptance Scenarios**:

1. **Given** I am logged in as an 'admin', **When** I navigate to the Admin Dashboard, **Then** I should see a DataTable of all registered users with their roles.
2. **Given** I am logged in as an 'admin', **When** I navigate to the Admin Dashboard, **Then** I should be able to toggle a user's role between 'lawyer' and 'admin'.
3. **Given** I am logged in as an 'admin', **When** I navigate to the Audit Logs section, **Then** I should see a chronological list of system activities (e.g., login, file uploads).

---

### User Story 2 - Legal Playbook & Golden Rules Management (Priority: P1)

As an Administrator, I want to manage the firm's legal guidelines by uploading playbook documents and defining "Golden Rules" in a text format so that the AI assistant can provide responses based on the most up-to-date firm-wide standards.

**Why this priority**: This is the core functionality for firm-wide consistency in AI-driven legal assistance. Without the playbook and rules, the AI lacks the specific context required for firm standards.

**Independent Test**: Can be tested by uploading a PDF/Docx file to the 'playbook' section and saving a text snippet to the 'Golden Rules' section, then verifying both are stored correctly.

**Acceptance Scenarios**:

1. **Given** I am on the Playbook Management page, **When** I upload a PDF or Docx file, **Then** it should be stored in the firm's document storage as a new version.
2. **Given** I am on the Playbook Management page, **When** I update the 'Golden Rules' text field and save, **Then** the updated rules should be persisted.

---

### User Story 3 - AI Context Integration (Priority: P2)

As a Lawyer, I want the AI assistant to provide legal analysis that incorporates both the uploaded Legal Playbook and the firm's 'Golden Rules' so that the advice I receive is aligned with both detailed procedures and high-level firm principles.

**Why this priority**: Enhances the quality and relevance of AI responses by providing multi-layered context.

**Independent Test**: Can be tested by asking the AI a question related to a specific rule in the Golden Rules and a procedural detail in the Playbook and verifying both are reflected in the response.

**Acceptance Scenarios**:

1. **Given** a Legal Playbook is uploaded and Golden Rules are defined, **When** I interact with the AI assistant, **Then** the response should explicitly or implicitly reflect standards from both sources, using source names (e.g., "According to the Playbook...") for citations.
2. **Given** a conflict between a Golden Rule and a Playbook detail, **When** the AI generates a response, **Then** it MUST highlight this contradiction to the user.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Admin Dashboard MUST include a DataTable to display and filter system users and their assigned roles.
- **FR-002**: System MUST allow Admins to update user roles directly from the Dashboard.
- **FR-003**: System MUST maintain an Audit Trail by logging key events (e.g., User Login, User Creation, Role Update, Playbook Upload) into a 'logs' table.
- **FR-004**: System MUST provide an interface for Admins to upload new versions of the firm's Legal Playbook file (PDF/Docx formats) and maintain a history of previous uploads.
- **FR-005**: System MUST provide a text-based interface for Admins to manage 'Golden Rules' that apply across the firm.
- **FR-006**: AI Logic MUST retrieve and utilize both the most recent Legal Playbook file and the current 'Golden Rules' text as context for generating responses.
- **FR-007**: AI Assistant MUST include explicit citations in its responses by referencing the source name (e.g., "Per Golden Rules...", "According to Playbook...").
- **FR-008**: AI Logic MUST detect and flag explicit conflicts between Golden Rules and Playbook content in its response to the user.
- **FR-009**: Users with 'lawyer' roles MUST NOT have write access to the Admin Dashboard or Playbook Management features.
- **FR-010**: Audit logs MUST be retained for 90 days before automated archival or deletion.

### Key Entities *(include if feature involves data)*

- **User**: System user with `id`, `email`, `role`, and `metadata`.
- **Log**: Audit trail entry with `id`, `user_id`, `event_type`, `description`, and `timestamp`.
- **Playbook**: Configuration entity with `id`, `file_url` (link to storage), `version_number`, `created_at`, and `golden_rules` (text).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Admins can view and filter 100% of registered users in the dashboard within 2 seconds of page load.
- **SC-002**: All critical administrative actions (e.g., role changes, playbook updates) are logged in the Audit Trail with 100% accuracy.
- **SC-003**: AI Assistant responses successfully reference context from both the Playbook and Golden Rules in relevant queries using explicit source names.
- **SC-004**: Unauthorized access to the Admin Dashboard (by non-admin users) is blocked with 100% reliability.
