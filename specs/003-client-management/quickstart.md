# Quickstart: Client & Case Management

## 1. Database Setup
Run the following SQL in the Supabase SQL Editor:
- **`clients` Table**: Schema and Case ID generation trigger (see `data-model.md`).
- **`documents` Table**: Schema (see `data-model.md`).
- **RLS Policies**: Enable and apply policies for both tables.

## 2. Storage Setup
- Create a **Private Bucket** named `client-vaults` in the Supabase Storage Dashboard.
- Ensure only `authenticated` users can read/write based on folder structure (`client_id`).

## 3. Environment Variables
No new environment variables required. Uses existing Supabase configuration.

## 4. Run Locally
```bash
npm run dev
```

## 5. Verification Flow
### Lawyer Onboarding
1. Sign in as a Lawyer.
2. Navigate to `/clients/new`.
3. Submit a new client (e.g., "John Doe", "Corporate").
4. Verify the client appears in your list with a Case ID (e.g., `[yourname]-XXXX`).

### Document Upload
1. Open a Client's Vault (`/clients/[id]/vault`).
2. Upload a PDF/DOCX/TXT file with a category (e.g., "Contract").
3. Verify the document is stored in the `client-vaults` bucket under the correct client ID folder.

### Admin Oversight
1. Sign in as an Admin.
2. Navigate to `/admin/clients`.
3. Verify all clients from all lawyers are visible.
4. Test the server-side search by both Client Name and Lawyer Name.
5. Test editing client details via the Admin modal.
