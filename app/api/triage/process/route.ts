import { NextResponse } from "next/server";
import { triageDocument } from "@/lib/ai/triage-service";
import { extractObligations } from "@/lib/ai/extractor";

export async function POST(request: Request) {
  try {
    const { documentId } = await request.json();

    if (!documentId) {
      return NextResponse.json({ error: "Document ID is required" }, { status: 400 });
    }

    // 1. Run Triage (Classification)
    const triageResult = await triageDocument(documentId);

    // 2. Run Extraction (Identify Obligations)
    const extractedObligations = await extractObligations(documentId);

    return NextResponse.json({ 
      success: true, 
      result: triageResult,
      extractedCount: extractedObligations.length 
    });
  } catch (error: any) {
    console.error("Triage API Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
