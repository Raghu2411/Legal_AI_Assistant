# Quickstart: Admin CRUD Console Validation

## Prerequisites
1. **Supabase Setup**:
   - Create `logs` table.
   - Create `playbooks` table.
   - Create `playbooks` storage bucket.
   - Ensure RLS is enabled on all tables.
2. **Environment Variables**:
   - `GROQ_API_KEY` must be set.
   - Supabase keys must be configured.

## Manual Validation Steps

### 1. Default Role Check
- Sign up a new user via `/login`.
- Verify in Supabase Dashboard that the user in `profiles` has `role = 'lawyer'`.

### 2. Admin Route Protection
- Log in as a 'lawyer'.
- Attempt to navigate to `/admin`.
- **Expected**: Redirected back to `/dashboard`.

### 3. User Oversight
- Log in as an 'admin'.
- Navigate to `/admin/users`.
- Change a lawyer's role to 'admin'.
- **Expected**: The update is reflected in the table and the `profiles` table.

### 4. Audit Trail
- Perform an action (e.g., login, role update).
- Navigate to `/admin/logs`.
- **Expected**: The action is logged with correct details.

### 5. Playbook Management
- Navigate to `/admin/playbook`.
- Upload a PDF and save "Golden Rules" text.
- **Expected**: File appears in Supabase storage; text appears in `playbooks` table.

### 6. AI Context Integration
- Ask the AI a question related to the "Golden Rules".
- **Expected**: AI references the rule and cites "Per Golden Rules...".

### 7. Lawyer Deletion (Integrity)
- Delete a lawyer via Admin interface (or simulated SQL).
- **Expected**: Clients/Docs originally assigned to that lawyer are now assigned to the Admin user.
