import { createClient } from "@/lib/supabase/server";
import { logDraftingAction } from "@/lib/ai/drafting-orchestrator";
import { processDocument } from "@/lib/ai/vector-service";
import { Packer, Document, Paragraph, TextRun } from "docx";

export async function POST(req: Request) {
  try {
    const { htmlContent, session, clientName } = await req.json();
    
    const supabase = createClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return new Response("Unauthorized", { status: 401 });
    }

    // 1. Convert HTML to plain text or simple DOCX
    // For MVP, we'll create a simple DOCX using the 'docx' library
    // Stripping HTML tags for a simple text-based DOCX
    const plainText = htmlContent.replace(/<[^>]*>?/gm, '\n').replace(/\n\s*\n/g, '\n\n').trim();
    
    const doc = new Document({
      sections: [{
        properties: {},
        children: [
          new Paragraph({
            children: [
              new TextRun({
                text: session.docName,
                bold: true,
                size: 32,
              }),
            ],
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: `Client: ${clientName}`,
                italics: true,
              }),
            ],
          }),
          new Paragraph({
            children: [
              new TextRun({
                text: `Generated on: ${new Date().toLocaleDateString()}`,
                italics: true,
              }),
            ],
          }),
          new Paragraph({ text: "" }), // Spacer
          ...plainText.split('\n').map((line: string) => new Paragraph({
            children: [new TextRun(line)],
          })),
        ],
      }],
    });

    const buffer = await Packer.toBuffer(doc);
    const fileName = `${session.docName.replace(/\s+/g, '_')}_${Date.now()}.docx`;
    const filePath = `${session.clientId}/${fileName}`;

    // 2. Upload to Supabase Storage
    const { data: storageData, error: storageError } = await supabase
      .storage
      .from('client-vaults')
      .upload(filePath, buffer, {
        contentType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      });

    if (storageError) throw storageError;

    // 3. Create database record in 'documents' table
    const { data: docRecord, error: docError } = await supabase
      .from('documents')
      .insert({
        client_id: session.clientId,
        file_name: session.docName,
        file_url: filePath,
        doc_type: 'Contract',
        uploaded_by: user.id,
        is_draft: true,
        draft_metadata: {
          docType: session.docType,
          session_id: session.startTime,
        },
        vector_status: 'Pending'
      })
      .select()
      .single();

    if (docError) throw docError;

    // 4. Log 'DRAFTING_FINALIZE'
    await logDraftingAction(user.id, session.clientId, 'DRAFTING_FINALIZE', {
      document_id: docRecord.id,
      document_name: session.docName,
    });

    // 5. Trigger RAG indexing (Async)
    // processDocument requires (sourceId, clientId, text)
    processDocument(docRecord.id, session.clientId, plainText).catch(err => {
      console.error("Error in background RAG indexing:", err);
    });

    return new Response(JSON.stringify({ 
      success: true, 
      documentId: docRecord.id,
      filePath: storageData.path 
    }), {
      status: 200,
      headers: { 'Content-Type': 'application/json' }
    });
  } catch (error: any) {
    console.error("Finalize error:", error);
    return new Response(JSON.stringify({ error: error.message }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
