# Quickstart: Policy Evolution Studio

## Prerequisites
- Python 3.11 with `python-docx` installed.
- Supabase project with `playbooks`, `golden_rules`, and `version_history` tables initialized.

## Setup Steps

1. **Evolution Studio UI**:
   - Access the Admin Dashboard and navigate to the "Evolution Studio" tab.
   - Upload the initial "Standard NDA Playbook" (JSON format or via existing Playbook importer).

2. **Compliance Standard Upload**:
   - Click "Upload Standard" and select a regulatory PDF/DOCX.
   - Wait for the "Gap Analysis" notification to complete.

3. **Approval Workflow**:
   - Review the AI-generated suggestions in the "Evolution Studio" side-by-side view.
   - Use checkboxes to select improvements and click "Approve Selected".

4. **Verify Generation & Sync**:
   - Confirm a new Playbook version exists in the "Version History".
   - Check the `documents` table to verify the new Playbook DOCX has been registered.
   - Verify that the `embeddings` table contains new entries corresponding to the generated Playbook ID.

5. **Instant Rule Update**:
   - Modify a Golden Rule in the "Evolution Studio".
   - Verify the `golden_rules` table reflects the update with a new timestamp and version immediately.

6. **Accountability Check**:
   - View the "Audit Trail" tab to see the "Before" and "After" state of the modified Golden Rule.
