# Data Model: Auth & RBAC Setup

## Profiles Table

Represents user metadata and role, linked to Supabase Auth `auth.users`.

- **id**: `uuid` (Primary Key, References `auth.users.id`)
- **full_name**: `text` (Required)
- **role**: `user_role` (Required, Default: 'lawyer')
- **created_at**: `timestamp with time zone` (Default: `now()`)
- **updated_at**: `timestamp with time zone` (Default: `now()`)

### Types/Enums

```sql
CREATE TYPE user_role AS ENUM ('admin', 'lawyer');
```

### Relationships

- `profiles.id` -> `auth.users.id` (1:1 relationship)

### Validation Rules

- `full_name` MUST NOT be empty.
- `role` MUST be one of the allowed enum values ('admin', 'lawyer').
- `updated_at` MUST be updated automatically via trigger.

### State Transitions

- **User Signup**: Trigger `on_auth_user_created()` -> Create `profiles` record.
- **Role Update**: Only 'admin' users can update a profile's `role`.
