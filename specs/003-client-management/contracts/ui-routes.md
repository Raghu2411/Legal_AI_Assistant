# UI Routes: Client & Case Management

## Lawyer Dashboard Routes
- **`/dashboard`**: Updated to include a "Clients" overview or link to `/clients`.
- **`/clients`**: Main client list for the logged-in lawyer.
- **`/clients/new`**: Client onboarding form.
- **`/clients/[id]`**: Client detail view (Overview + Navigation to Vault).
- **`/clients/[id]/vault`**: Document Vault for the specific client.

## Admin Routes
- **`/admin/clients`**: Firm-wide client management with server-side search (Lawyer & Client names).
- **`/admin/logs`**: Existing audit log, now tracking client/document events.

## Route Protection
- All `/clients` routes require `authenticated` status and `lawyer` OR `admin` role.
- `/admin` routes strictly require `admin` role.
- RLS ensures lawyers cannot access `/clients/[id]` if they don't own the client.
