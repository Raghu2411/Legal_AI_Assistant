# UI Routes: Admin CRUD Console

## Admin Layout
**Path**: `/admin`
**Protection**: Requires `admin` role in `profiles` table.
**Navigation**:
- Dashboard Overview (`/admin`)
- User Oversight (`/admin/users`)
- Audit Trail (`/admin/logs`)
- Playbook Management (`/admin/playbook`)

---

## Route: Dashboard Overview
**Path**: `/admin`
**Description**: High-level summary of system status.
**Components**:
- Total Users Count
- Recent Activity Feed (last 5 logs)
- Active Playbook Version

---

## Route: User Oversight
**Path**: `/admin/users`
**Description**: Centralized user management.
**UI Elements**:
- shadcn/ui DataTable
- Columns: Email, Role, Created At, Actions (Toggle Role)
**Actions**:
- `toggleRole(userId, currentRole)`: Updates `profiles.role`.

---

## Route: Audit Trail
**Path**: `/admin/logs`
**Description**: History of system events.
**UI Elements**:
- shadcn/ui DataTable
- Columns: User, Event Type, Description, Metadata, Timestamp
- Filter by: Event Type, User Email

---

## Route: Playbook Management
**Path**: `/admin/playbook`
**Description**: Management of firm-wide rules.
**UI Elements**:
- Hybrid Form:
  - File Upload (PDF/Docx) -> Supabase Storage
  - Textarea (Golden Rules) -> Postgres
- Version History Table (list of previous versions)
**Actions**:
- `updatePlaybook(formData)`: Stores file, updates DB record, increments version.
