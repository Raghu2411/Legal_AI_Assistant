# Feature Specification: Auth & RBAC Setup

**Feature Branch**: `001-auth-rbac-setup`
**Created**: 2026-02-25
**Status**: Draft
**Input**: User description: "Create technical specifications for Step 1. - Framework: Next.js 14+ App Router, Tailwind CSS, shadcn/ui. - Auth: Supabase Auth with Email/Password. - Database Schema: A 'profiles' table linked to 'auth.users' containing: id (uuid), full_name (text), and role (enum: admin, lawyer). - Middleware: Implement Next.js middleware to handle role-based redirection. /admin routes for admins, /dashboard and other modules for lawyers. - UI: A login page with a shadcn/ui card component. - Env: Ensure the system uses NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY."

## Clarifications

### Session 2026-02-25
- Q: What is the expected behavior after a successful login for different roles? → A: Redirect to role-specific workspace (e.g., /admin or /dashboard)
- Q: How should the system handle users who exist in Supabase Auth but are missing a corresponding entry in the 'profiles' table? → A: Redirect to an "Access Denied / Setup Required" page
- Q: How should the system handle an authenticated user attempting to access a route for which they lack sufficient permissions (e.g., a Lawyer trying to access /admin)? → A: Redirect to their role-specific workspace (e.g., /dashboard)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Secure Login (Priority: P1)

As a user (Admin or Lawyer), I want to log into the application using my email and password so that I can access my respective workspace safely.

**Why this priority**: Core entry point. Without authentication, no other features can be accessed.

**Independent Test**: Manual verification: Enter valid credentials on the /login page, click login, and verify redirection to the correct role-based landing page.

**Acceptance Scenarios**:

1. **Given** an unauthenticated user, **When** they navigate to any protected route (e.g., /admin, /dashboard), **Then** they are redirected to /login.
2. **Given** a user with 'admin' role, **When** they login successfully, **Then** they are redirected to /admin.
3. **Given** a user with 'lawyer' role, **When** they login successfully, **Then** they are redirected to /dashboard.
4. **Given** invalid credentials, **When** the user attempts to login, **Then** they see an error message and remain on the /login page.

---

### User Story 2 - Role-Based Access Control (Priority: P2)

As an Admin, I want to ensure that only I can access administrative routes, and as a Lawyer, I want to ensure I am directed to my workspace.

**Why this priority**: Essential for data isolation and security as per Constitution Principle IV.

**Independent Test**: Manual verification: Attempt to access /admin as a user with the 'lawyer' role and verify that access is denied/redirected to /dashboard.

**Acceptance Scenarios**:

1. **Given** a logged-in user with role 'lawyer', **When** they attempt to visit /admin, **Then** they are redirected back to /dashboard.
2. **Given** a logged-in user with role 'admin', **When** they attempt to visit /dashboard, **Then** they are allowed access (Admins have global access).

---

### Edge Cases

- **Session Expiry**: User session expires while on a protected page. System MUST redirect to /login on next interaction.
- **Direct Route Access**: User attempts to visit a nested protected route directly via URL. Middleware MUST intercept and redirect to /login if unauthenticated.
- **Missing Profile**: User exists in Auth but missing 'profiles' entry. System MUST redirect to an "Access Denied / Setup Required" page.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a login interface using a shadcn/ui Card component.
- **FR-002**: System MUST authenticate users via email and password using Supabase Auth.
- **FR-003**: System MUST store user profile information (full_name, role) in a 'profiles' table linked to 'auth.users'.
- **FR-004**: System MUST implement Next.js Middleware to protect all routes except /login.
- **FR-005**: System MUST redirect users based on their 'role' metadata after a successful login.
- **FR-006**: System MUST ensure 'admin' role has access to /admin routes.
- **FR-007**: System MUST ensure 'lawyer' role has access to /dashboard and related modules, but NOT /admin.

### Key Entities

- **Profile**: Represents user metadata and role.
  - `id`: Unique identifier (UUID, links to `auth.users.id`)
  - `full_name`: User's display name (text)
  - `role`: Access level (enum: 'admin', 'lawyer')

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Unauthenticated users are redirected to /login 100% of the time when accessing protected routes.
- **SC-002**: Redirection based on role (Admin -> /admin, Lawyer -> /dashboard) occurs immediately upon successful login.
- **SC-003**: Access to /admin is restricted to users with the 'admin' role only.
- **SC-004**: Users see a clear error message within 1 second of an invalid login attempt.
