# Feature Specification: Policy Evolution Studio

**Feature Branch**: `009-policy-evolution`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "Create technical specs for Step 9. - UI: Evolution Studio with checkbox-based approval for AI suggestions. - Generation: Implement a server-side document generator for updating Playbook files. - Integration: Ensure updated Golden Rules are instantly available to the Module 2 and Module 5 AI prompts. - Data: Log all changes to a 'version_history' table for full firm accountability."

## Clarifications

### Session 2026-03-22
- Q: Should access to the Evolution Studio be strictly limited to the 'admin' role, or can senior 'lawyers' be granted permission to review and suggest (but not approve) changes? → A: Strictly 'admin' role only (Standard)
- Q: Should the Evolution Studio allow Admins to rollback to a previous policy state directly from the Version History, or is history intended to be read-only for audit purposes? → A: One-click rollback enabled (High Agility)
- Q: How should the system handle concurrent policy edits in the Evolution Studio? If two Admins are reviewing suggestions simultaneously, how should a conflict be resolved? → A: Last Write Wins with notification (Simple)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Compliance Auditing & Suggestions (Priority: P1)

Admins need to ensure firm policies stay current with changing laws. They upload a new Compliance Standard document, and the AI audits the existing Playbook and Golden Rules against it, highlighting gaps and suggesting specific text updates.

**Why this priority**: foundational capability for policy evolution; without auditing, there are no suggestions to approve.

**Independent Test**: Admin uploads a PDF containing a single new regulation. System identifies the missing clause in the current Playbook and suggests a specific update.

**Acceptance Scenarios**:

1. **Given** an existing "Standard NDA Playbook", **When** Admin uploads "New State NDA Regulation 2026", **Then** the system displays a list of gaps found.
2. **Given** identified gaps, **When** AI generates suggestions, **Then** each suggestion must link to the specific section of the Compliance Standard.

---

### User Story 2 - Evolution Studio Review (Priority: P1) 🎯 MVP

Admins require a precise way to control which AI suggestions become firm law. The Evolution Studio provides a side-by-side view of current policy vs. suggested change with a checkbox for selective approval.

**Why this priority**: Essential for "Admin Sovereignty" and "Checkbox-based approval" requirement.

**Independent Test**: Admin selects 2 out of 3 suggested Golden Rule updates using checkboxes. Only those 2 are marked for implementation.

**Acceptance Scenarios**:

1. **Given** 5 AI suggestions, **When** Admin checks 3 and clicks 'Approve Selected', **Then** the system only processes those 3 updates.
2. **Given** a suggested change, **When** Admin clicks 'View Source', **Then** the relevant passage from the Compliance Standard is highlighted.

---

### User Story 3 - Playbook Generation & RAG Sync (Priority: P2)

Once changes are approved, the Playbook file must be updated and re-indexed so the latest logic is available in the vector database.

**Why this priority**: Ensures the "Legal Playbook" remains the single source of truth for the RAG pipeline (Constitution XXXVI).

**Independent Test**: Admin approves a Playbook update. System generates a new DOCX file. Database records confirm the file is stored and `processDocument` is triggered.

**Acceptance Scenarios**:

1. **Given** approved Playbook changes, **When** the generation process starts, **Then** a new versioned DOCX is saved to Supabase Storage.
2. **Given** a new Playbook file, **When** generation completes, **Then** the RAG indexing pipeline (mxbai-embed-large-v1) MUST be invoked with the new document ID.

---

### User Story 4 - Instant Golden Rule Propagation (Priority: P1)

Golden Rules must be updated in the database immediately so that the next execution of any AI prompt uses the latest logic without delay.

**Why this priority**: Critical requirement for "instant availability" and "Golden Rule Precedence" (Constitution XIX).

**Independent Test**: Admin updates a Golden Rule in the studio. Database records confirm the update is committed with a new timestamp and version.

**Acceptance Scenarios**:

1. **Given** an approved Golden Rule change, **When** the Admin clicks 'Save', **Then** the `golden_rules` table is updated atomically.
2. **Given** a rule update, **When** the system fetches rules for a new operation, **Then** it receives the version updated < 1 second ago.

---

### User Story 5 - Accountability & Version History (Priority: P2)

The firm requires a transparent record of all policy changes for internal audits and liability protection.

**Why this priority**: Mandatory requirement for "full firm accountability" and Constitution XXXVIII.

**Independent Test**: Admin views the 'Version History' tab. They see a list of all changes, including the specific timestamp, user, and "Before vs. After" text comparison.

**Acceptance Scenarios**:

1. **Given** a completed policy update, **When** Admin views Version History, **Then** an entry exists showing exactly which clauses were modified.
2. **Given** a history entry, **When** clicked, **Then** it displays a diff-style view of the changes.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Evolution Studio MUST provide a dashboard for uploading external Compliance Standards (PDF/DOCX).
- **FR-002**: AI MUST compare uploaded standards against existing `playbooks` and `golden_rules` to identify contradictions or omissions.
- **FR-003**: System MUST present AI suggestions in a tabular/list view with checkboxes for approval.
- **FR-004**: System MUST include a server-side document generator capable of producing DOCX files from structured Playbook data.
- **FR-005**: System MUST increment the `version` field in the `playbooks` table upon every generation.
- **FR-006**: System MUST update the `golden_rules` table in real-time upon Admin approval.
- **FR-007**: System MUST log all modifications (User ID, Timestamp, Field, Old Value, New Value) to a `version_history` table.
- **FR-008**: AI prompts in Module 2 and Module 5 MUST fetch Golden Rules directly from the database for every execution to ensure 0-latency synchronization.
- **FR-009**: System MUST trigger the `processDocument` function for the new Playbook version immediately after generation.
- **FR-010**: Access to the Evolution Studio MUST be strictly limited to users with the 'admin' role.
- **FR-011**: System MUST allow Admins to restore any previous version of a Golden Rule or Playbook directly from the Version History.
- **FR-012**: Rolling back a Playbook MUST trigger a mandatory RAG re-indexing for the restored version.
- **FR-013**: Concurrent edits in the Evolution Studio MUST follow a 'Last Write Wins' model, with a notification sent to users whose changes are overwritten.

### Key Entities *(include if feature involves data)*

- **ComplianceStandard**: Represents uploaded external regulatory documents.
- **PolicySuggestion**: A transient entity representing AI-proposed changes to Playbooks or Golden Rules.
- **PlaybookVersion**: Tracks the evolution of the global Legal Playbook file.
- **GoldenRule**: The specific database-driven rules for high-speed AI reasoning.
- **VersionHistory**: The immutable audit trail for all policy changes.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Policy suggestion generation completes in under 30 seconds for a 50-page compliance document.
- **SC-002**: 100% of approved Golden Rule changes are reflected in Module 5 within 500ms of database commit.
- **SC-003**: The server-side generator produces valid DOCX files that open in Microsoft Word without corruption.
- **SC-004**: Every policy change since feature launch has a corresponding entry in the `version_history` table.
- **SC-005**: Users can view a "Before vs After" diff for any historical change in under 2 seconds.
- **SC-006**: Rolling back a Golden Rule restores its state across the system in under 1 second.

## Assumptions & Defaults

- **Assumption 1**: The server-side generator will focus on DOCX first, as it allows for easier manual cleanup by legal staff.
- **Assumption 2**: Playbooks are stored as structured JSON/Markdown in the database to facilitate the "diff" view and document generation.
- **Assumption 3**: Compliance standards are temporary "audit inputs" and do not need to be indexed for long-term RAG unless explicitly requested.
- **Default 1**: The system will retain all historical versions indefinitely to satisfy the "full firm accountability" requirement.
