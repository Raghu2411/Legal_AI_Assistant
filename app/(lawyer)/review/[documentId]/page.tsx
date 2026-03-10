import { createClient } from "@/lib/supabase/server";
import { notFound, redirect } from "next/navigation";
import { ReviewStudio } from "@/components/review/review-studio";

export default async function ReviewPage({ params }: { params: { documentId: string } }) {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  // Fetch document details
  const { data: doc } = await supabase
    .from('documents')
    .select('*, clients(name, auto_case_id)')
    .eq('id', params.documentId)
    .single();

  if (!doc) {
    notFound();
  }

  // Fetch all RiskAnalyses for history
  const { data: scanHistory } = await supabase
    .from('risk_analyses')
    .select('id, created_at, status')
    .eq('document_id', params.documentId)
    .order('created_at', { ascending: false });

  // Fetch latest RiskAnalysis if any
  const { data: latestAnalysis } = await supabase
    .from('risk_analyses')
    .select('*, clause_analyses(*)')
    .eq('document_id', params.documentId)
    .order('created_at', { ascending: false })
    .limit(1)
    .single();

  // Fetch document text from embeddings
  const { data: chunks } = await supabase
    .from('embeddings')
    .select('content, metadata')
    .eq('document_id', params.documentId);

  const fullText = (chunks || [])
    .sort((a, b) => (a.metadata?.chunk_index || 0) - (b.metadata?.chunk_index || 0))
    .map(c => c.content)
    .join("\n\n");

  return (
    <ReviewStudio 
      document={doc}
      initialAnalysis={latestAnalysis}
      documentText={fullText}
      scanHistory={scanHistory || []}
    />
  );
}
