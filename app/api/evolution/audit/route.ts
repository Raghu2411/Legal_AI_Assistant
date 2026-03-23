import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";
import { runGapAnalysis } from "@/lib/ai/evolution-orchestrator";

export async function POST(request: Request) {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  // Check admin role
  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", user.id)
    .single();

  if (profile?.role !== 'admin') {
    return NextResponse.json({ error: "Forbidden: Admin access required" }, { status: 403 });
  }

  try {
    const formData = await request.formData();
    const file = formData.get("file") as File;
    const name = formData.get("name") as string;

    if (!file) {
      return NextResponse.json({ error: "No file uploaded" }, { status: 400 });
    }

    // 1. Upload to Supabase Storage
    const fileName = `${Date.now()}_${file.name}`;
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from("compliance-standards")
      .upload(fileName, file);

    if (uploadError) throw uploadError;

    // 2. Register in DB
    const { data: standard, error: dbError } = await supabase
      .from("compliance_standards")
      .insert({
        name: name || file.name,
        storage_path: uploadData.path,
        uploaded_by: user.id,
      })
      .select()
      .single();

    if (dbError) throw dbError;

    // 3. Trigger Gap Analysis
    console.log(`[AuditRoute] Running runGapAnalysis for standard ${standard.id}`);
    const analysisResult = await runGapAnalysis(standard.id, user.id);
    console.log(`[AuditRoute] runGapAnalysis completed. Suggestions: ${analysisResult.suggestionCount}`);

    return NextResponse.json({ 
        message: "Upload and gap analysis successful.", 
        standardId: standard.id,
        suggestionCount: analysisResult.suggestionCount
    });

  } catch (error: any) {
    console.error("Audit Upload Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
