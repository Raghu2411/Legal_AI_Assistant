-- Add DELETE policy for obligations
CREATE POLICY "Lawyers can delete obligations for their clients" ON public.obligations
    FOR DELETE USING (
        EXISTS (
            SELECT 1 FROM public.clients
            WHERE id = obligations.client_id
            AND (lawyer_id = auth.uid() OR (SELECT role FROM public.profiles WHERE id = auth.uid()) = 'admin')
        )
    );
