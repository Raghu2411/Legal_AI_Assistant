# Quickstart: Auth & RBAC Setup

## Prerequisites

- [Node.js 18+](https://nodejs.org/)
- [Supabase Account](https://supabase.com/)
- [Git](https://git-scm.com/)

## Environment Setup

1.  Clone the repository and install dependencies:
    ```bash
    npm install
    ```
2.  Create a `.env.local` file with your Supabase credentials:
    ```env
    NEXT_PUBLIC_SUPABASE_URL=your-project-url
    NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
    ```

## Database Initialization

Run the following SQL in your Supabase SQL Editor:

```sql
-- Create types and table
CREATE TYPE user_role AS ENUM ('admin', 'lawyer');

CREATE TABLE profiles (
  id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  full_name TEXT NOT NULL,
  role user_role NOT NULL DEFAULT 'lawyer',
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Public profiles are viewable by everyone." ON profiles
  FOR SELECT USING (true);

CREATE POLICY "Users can insert their own profile." ON profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

CREATE POLICY "Users can update their own profile." ON profiles
  FOR UPDATE USING (auth.uid() = id);

-- Trigger for auto-profile creation
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
  INSERT INTO public.profiles (id, full_name, role)
  VALUES (new.id, new.raw_user_meta_data->>'full_name', 'lawyer');
  RETURN new;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE PROCEDURE public.handle_new_user();
```

## Running the Application

1.  Start the development server:
    ```bash
    npm run dev
    ```
2.  Navigate to `http://localhost:3000/login`.
3.  Sign up a new user to trigger profile creation.
4.  Verify redirection based on the default 'lawyer' role.
