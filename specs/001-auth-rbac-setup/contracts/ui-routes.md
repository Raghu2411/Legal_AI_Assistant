# UI Routes: Auth & RBAC Setup

## Public Routes

- `/login`: The main entry point for unauthenticated users.

## Protected Routes

- `/dashboard`: Primary workspace for users with the 'lawyer' role.
- `/admin`: Primary workspace for users with the 'admin' role.

## Route Protection Logic

- **Unauthenticated**: All requests to `/dashboard` or `/admin` MUST redirect to `/login`.
- **Authenticated (Lawyer)**: Requests to `/admin` MUST redirect to `/dashboard`.
- **Authenticated (Admin)**: Requests to `/dashboard` are allowed (Admins have global access).
- **Missing Profile**: If a user exists in Auth but has no `profiles` entry, redirect to `/access-denied`.
