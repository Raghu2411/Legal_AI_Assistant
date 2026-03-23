-- Add suggestion_id to version_history to link actions to suggestions
ALTER TABLE public.version_history
ADD COLUMN IF NOT EXISTS suggestion_id UUID REFERENCES public.policy_suggestions(id) ON DELETE SET NULL;
