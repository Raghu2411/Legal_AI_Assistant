-- Update playbooks table with versioning and metadata
ALTER TABLE public.playbooks 
ADD COLUMN IF NOT EXISTS version INT DEFAULT 1,
ADD COLUMN IF NOT EXISTS last_updated_by UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS last_updated_at TIMESTAMPTZ DEFAULT now();

-- Update golden_rules table with versioning and metadata
ALTER TABLE public.golden_rules
ADD COLUMN IF NOT EXISTS version INT DEFAULT 1,
ADD COLUMN IF NOT EXISTS last_updated_by UUID REFERENCES auth.users(id),
ADD COLUMN IF NOT EXISTS last_updated_at TIMESTAMPTZ DEFAULT now();

-- Create compliance_standards table
CREATE TABLE IF NOT EXISTS public.compliance_standards (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    uploaded_by UUID NOT NULL REFERENCES auth.users(id),
    uploaded_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS on compliance_standards
ALTER TABLE public.compliance_standards ENABLE ROW LEVEL SECURITY;

-- Create policy_suggestions table (Transient)
CREATE TABLE IF NOT EXISTS public.policy_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    standard_id UUID REFERENCES public.compliance_standards(id) ON DELETE CASCADE,
    target_type TEXT NOT NULL, -- 'playbook', 'golden_rule'
    target_id UUID NOT NULL,
    current_text TEXT,
    suggested_text TEXT NOT NULL,
    rationale TEXT,
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'approved', 'rejected'
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS on policy_suggestions
ALTER TABLE public.policy_suggestions ENABLE ROW LEVEL SECURITY;

-- Create version_history table (Immutable)
CREATE TABLE IF NOT EXISTS public.version_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_type TEXT NOT NULL, -- 'playbook', 'golden_rule'
    entity_id UUID NOT NULL,
    field TEXT,
    old_value JSONB,
    new_value JSONB,
    change_type TEXT NOT NULL DEFAULT 'update', -- 'update', 'rollback', 'generation'
    user_id UUID NOT NULL REFERENCES auth.users(id),
    timestamp TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS on version_history
ALTER TABLE public.version_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies: Admin-only access for evolution-related tables

-- compliance_standards
CREATE POLICY "Admins can manage compliance_standards"
    ON public.compliance_standards
    FOR ALL
    USING ((SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
    WITH CHECK ((SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin');

-- policy_suggestions
CREATE POLICY "Admins can manage policy_suggestions"
    ON public.policy_suggestions
    FOR ALL
    USING ((SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
    WITH CHECK ((SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin');

-- version_history
CREATE POLICY "Admins can view version_history"
    ON public.version_history
    FOR SELECT
    USING ((SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin');

CREATE POLICY "Admins can insert version_history"
    ON public.version_history
    FOR INSERT
    WITH CHECK ((SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin');
