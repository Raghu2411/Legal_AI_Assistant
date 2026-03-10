-- Add current_content to documents table to store TipTap state
ALTER TABLE public.documents 
ADD COLUMN IF NOT EXISTS current_content text;
