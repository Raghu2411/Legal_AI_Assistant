# Quickstart: Smart Triage & Operations (Step 8)

## 1. Prerequisites
- `GROQ_API_KEY` configured with Llama 3.3.
- `obligations` table migrated to Supabase.
- Admin Golden Rules defined in the Playbook.

## 2. Testing the Triage Process
1. Upload a document to a Client Vault.
2. Trigger the Triage Scan from the Client Detail page.
3. Verify the classification (Standard vs Complex) appears in the Triage Queue.

## 3. Verifying Obligations
1. Open the Operations Dashboard.
2. Review the 'Pending' obligations extracted by the AI.
3. Click 'Confirm' to add the obligation to the Operations Calendar.
4. Verify the entry appears on the shadcn/ui Calendar.

## 4. Dual-Scope Compliance
1. Identify a clause that violates firm policy but passes regulatory standards.
2. Verify that the system flags the specific layer failure in the Compliance Sidebar.
