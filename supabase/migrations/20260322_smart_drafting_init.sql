-- Create activity_logs table
CREATE TABLE IF NOT EXISTS public.activity_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    client_id UUID REFERENCES public.clients(id) ON DELETE CASCADE,
    action_type TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS on activity_logs
ALTER TABLE public.activity_logs ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can see their own logs or admins can see all
CREATE POLICY "Users and admins can view activity logs"
    ON public.activity_logs
    FOR SELECT
    USING (
        auth.uid() = user_id 
        OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin'
    );

-- RLS Policy: Users can insert their own logs or admins can insert
CREATE POLICY "Users and admins can insert activity logs"
    ON public.activity_logs
    FOR INSERT
    WITH CHECK (
        auth.uid() = user_id 
        OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin'
    );

-- Extend documents table
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS is_draft BOOLEAN DEFAULT false,
ADD COLUMN IF NOT EXISTS draft_metadata JSONB DEFAULT '{}'::jsonb;

-- Ensure RLS covers new columns (existing policies should handle this if they are table-wide)
-- If specific column policies exist, they might need updates, but usually it's row-level.
