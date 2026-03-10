-- Create custom types for review studio
DO $$ BEGIN
    CREATE TYPE public.risk_status AS ENUM ('green', 'yellow', 'red');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE public.analysis_status AS ENUM ('pending', 'completed', 'failed');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- RiskAnalysis Table
CREATE TABLE IF NOT EXISTS public.risk_analyses (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES public.documents(id) ON DELETE CASCADE,
    created_at timestamptz DEFAULT now(),
    status public.analysis_status DEFAULT 'pending',
    raw_json jsonb,
    version int DEFAULT 1
);

-- ClauseAnalysis Table
CREATE TABLE IF NOT EXISTS public.clause_analyses (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    risk_analysis_id uuid NOT NULL REFERENCES public.risk_analyses(id) ON DELETE CASCADE,
    original_text text,
    risk_status public.risk_status NOT NULL,
    rationale text,
    suggested_rewrite text,
    user_overridden_status public.risk_status,
    user_override_rationale text,
    is_gap boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);

-- GoldenRule Table
CREATE TABLE IF NOT EXISTS public.golden_rules (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    admin_id uuid REFERENCES public.profiles(id) ON DELETE SET NULL,
    rule_text text NOT NULL,
    priority int DEFAULT 0,
    is_active boolean DEFAULT true,
    created_at timestamptz DEFAULT now()
);

-- Add review_status to documents table
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS review_status text DEFAULT 'uploaded';

-- Add a constraint to ensure review_status is valid
DO $$ BEGIN
    ALTER TABLE public.documents
    ADD CONSTRAINT documents_review_status_check
    CHECK (review_status IN ('uploaded', 'scanning', 'analyzed', 'reviewed'));
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Indexes for performance
CREATE INDEX IF NOT EXISTS risk_analyses_document_id_idx ON public.risk_analyses (document_id);
CREATE INDEX IF NOT EXISTS clause_analyses_risk_analysis_id_idx ON public.clause_analyses (risk_analysis_id);
CREATE INDEX IF NOT EXISTS golden_rules_is_active_idx ON public.golden_rules (is_active) WHERE is_active = true;
