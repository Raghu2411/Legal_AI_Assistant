import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";
import { exec } from "child_process";
import { promisify } from "util";
import fs from "fs/promises";
import path from "path";
import { processDocument } from "@/lib/ai/vector-service";
import { extractTextFromFile } from "@/lib/playbook/parser";

const execAsync = promisify(exec);

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
    const { playbookId } = await request.json();

    // 1. Fetch latest playbook content
    const { data: playbook, error: pError } = await supabase
      .from("playbooks")
      .select("*")
      .eq("id", playbookId)
      .single();

    if (pError || !playbook) throw new Error("Playbook not found");

    // 2. Prepare temp files
    const tmpDir = path.join(process.cwd(), "tmp");
    await fs.mkdir(tmpDir, { recursive: true });
    
    const inputPath = path.join(tmpDir, `input_${playbookId}.json`);
    const outputPath = path.join(tmpDir, `output_${playbookId}.docx`);

    await fs.writeFile(inputPath, JSON.stringify({
        version: playbook.version,
        status: playbook.status,
        last_updated: playbook.last_updated_at,
        sections: playbook.content.sections || []
    }));

    // 3. Execute Python Generator
    const scriptPath = path.join(process.cwd(), "lib/operations/docx-generator.py");
    await execAsync(`python "${scriptPath}" "${inputPath}" "${outputPath}"`);

    // 4. Upload to Supabase Storage
    const fileBuffer = await fs.readFile(outputPath);
    const fileName = `playbooks/${playbook.name}_v${playbook.version}.docx`;
    
    const { data: uploadData, error: uploadError } = await supabase.storage
      .from("playbook-artifacts")
      .upload(fileName, fileBuffer, {
        contentType: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        upsert: true
      });

    if (uploadError) throw uploadError;

    // 5. Update Playbook record with file path
    await supabase
      .from("playbooks")
      .update({
        file_path: uploadData.path,
        last_updated_at: new Date().toISOString()
      })
      .eq("id", playbookId);

    // 6. Trigger RAG Sync (US3 - T018)
    // We extract text from the generated DOCX and vectorize it
    const text = await extractTextFromFile(fileBuffer, playbook.name);
    
    // We run this in the background to avoid timing out the request
    processDocument(playbookId, null, text, 'playbooks').catch(err => {
      console.error("RAG Sync Background Error:", err);
    });

    // 7. Cleanup
    await fs.unlink(inputPath);
    await fs.unlink(outputPath);

    return NextResponse.json({ 
        message: "Playbook generated and RAG synchronization started.",
        filePath: uploadData.path
    });

  } catch (error: any) {
    console.error("Generation Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
