# Feature Specification: Smart Triage & Operations

**Feature Branch**: `008-smart-triage-operations`  
**Created**: 2026-03-22  
**Status**: Draft  
**Input**: User description: "Create technical specs for Step 8. - UI: Operations Dashboard with shadcn/ui Calendar and a Triage Queue table. - Database: Create 'obligations' table with a confirmation status workflow. - AI: Use Groq/Llama 3.3 for classification based on Golden Rules keywords and for dual-scope compliance flagging. - Logic: Implement the 'Confirm to Calendar' transition for extracted milestones."

## Clarifications

### Session 2026-03-22
- Q: If the AI misinterprets keywords during triage, should Admins/Lawyers have manual override capability for the 'Standard' vs 'Complex' classification? → A: Allow Manual Override: Admins/Lawyers can manually change 'Standard' to 'Complex' (or vice-versa), with a required reason.
- Q: How should the system handle obligations where the AI identifies a task but cannot find a specific date in the text? → A: TBD / Missing Flag: Mark the date as "TBD" and require manual input during the 'Confirm' step.
- Q: How should the system visually represent dual-scope compliance flagging (Admin vs. Regulatory)? → A: Distinct Status Icons: Show separate indicators (e.g., "Policy" and "Regulatory") for each layer.
- Q: How should the system handle AI-extracted obligations that a lawyer explicitly rejects during the verification step? → A: Soft Delete / Archive: Retain the record with a "Rejected" status for audit purposes, but hide it from the main dashboard.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Automated Document Triage (Priority: P1)

As a Lawyer or Admin, I want the system to automatically categorize uploaded documents based on our firm's "Golden Rules" so that I can immediately identify which files require senior expert attention and which are standard.

**Why this priority**: Essential for high-volume management and risk identification. It ensures complex files are never missed.

**Independent Test**: Upload a document containing specific Golden Rule keywords (e.g., "uncapped liability") and verify it is tagged as 'Complex' in the Triage Queue. Upload a standard NDA and verify it is tagged as 'Standard'. Test manual override by changing a 'Standard' flag to 'Complex' and verifying the mandatory reason field.

**Acceptance Scenarios**:

1. **Given** a new document is uploaded, **When** the triage process runs, **Then** it must be visible in the Triage Queue table with a classification status.
2. **Given** a document marked as 'Complex', **When** viewed in the dashboard, **Then** the system must highlight the specific Golden Rule keywords that triggered the classification.
3. **Given** an automated classification, **When** a user attempts to override it, **Then** the system MUST require a text-based reason and log the change to the audit trail.

---

### User Story 2 - Obligation Extraction & Verification (Priority: P2)

As a Lawyer, I want the AI to extract specific legal obligations (dates, tasks, milestones) from a document and present them for my manual confirmation so that I can ensure accuracy before they are finalized.

**Why this priority**: Directly supports Principle XXXII (Obligation Verification). Prevents incorrect dates or tasks from entering the firm's operational workflow.

**Independent Test**: Open a document in the Operations Dashboard, trigger obligation extraction, and verify that the system presents a list of "Draft" obligations with 'Confirm' and 'Reject' options.

**Acceptance Scenarios**:

1. **Given** extracted draft obligations, **When** I click 'Confirm' on an obligation, **Then** its status changes from 'Pending' to 'Confirmed'.
2. **Given** a confirmed obligation, **When** viewed in the dashboard, **Then** it must include an audit trail of who confirmed it and when.

---

### User Story 3 - Operational Calendar Management (Priority: P3)

As a Lawyer, I want to see all confirmed obligations on a unified calendar so that I can manage my upcoming deadlines and tasks effectively.

**Why this priority**: Provides the "transition to calendar" logic requested. Visualizes the output of the triage and verification process.

**Independent Test**: Confirm an obligation with a specific date and verify that a corresponding entry appears on the Operations Calendar for that date.

**Acceptance Scenarios**:

1. **Given** the Operations Dashboard is open, **When** I view the Calendar tab, **Then** I must see all confirmed obligations plotted on their respective due dates.
2. **Given** an obligation on the calendar, **When** I click on it, **Then** it must display the original document context and the confirmation details.

---

### Edge Cases

- **Classification Ambiguity**: How does the system handle documents that match both 'Standard' and 'Complex' criteria or neither? (Default to 'Complex' for safety).
- **Date Extraction Failure**: AI-identified obligations with no clear date MUST be marked as "TBD". These obligations remain in "Pending" status and CANNOT be confirmed until a valid date is manually provided by a Lawyer.
- **User Overrides**: What happens if a Lawyer wants to manually add an obligation that the AI missed?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide an Operations Dashboard with a Triage Queue (Table) and an Operations Calendar (shadcn/ui Calendar).
- **FR-002**: System MUST use Groq/Llama 3.3 to classify documents as 'Standard' or 'Complex' based on Admin-defined Golden Rules keywords. Users MUST be able to manually override this classification with a mandatory justification.
- **FR-003**: System MUST identify and flag dual-scope compliance issues (Admin policies vs. general regulatory standards) using distinct visual indicators for each layer.
- **FR-004**: System MUST store extracted obligations in an `obligations` table with status tracking (Pending, Confirmed, Rejected).
- **FR-005**: System MUST prevent any obligation from appearing on the Operations Calendar until it has been explicitly confirmed by a Lawyer.
- **FR-006**: System MUST maintain an immutable audit trail for every triage decision and obligation confirmation in the `activity_logs`.
- **FR-007**: Users MUST be able to manually edit or add obligations during the verification step.
- **FR-008**: System MUST support "TBD" dates for obligations, enforcing manual date entry before confirmation is permitted.

### Key Entities *(include if feature involves data)*

- **Obligation**: Represents a specific task, date, or milestone extracted from a document.
    - Attributes: `id`, `document_id`, `client_id`, `description`, `due_date`, `status`, `assigned_lawyer_id`, `confirmed_at`.
- **Activity Log**: Already exists, but must be extended/utilized for Triage and Obligation events.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Triage classification for a standard document (under 20 pages) completes in under 10 seconds.
- **SC-002**: 100% of confirmed obligations correctly appear on the Operations Calendar within 1 second of confirmation.
- **SC-003**: Admins can view a complete audit trail for any triage decision within 2 clicks from the main dashboard.
- **SC-004**: System correctly flags 90% of Golden Rule keyword matches in test "Complex" documents.
