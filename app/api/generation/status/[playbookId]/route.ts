import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

export async function GET(
  request: Request,
  { params }: { params: { playbookId: string } }
) {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const { playbookId } = params;

  try {
    const { data: playbook, error } = await supabase
      .from("playbooks")
      .select("vector_status, last_vectorized, version, last_updated_at")
      .eq("id", playbookId)
      .single();

    if (error) throw error;

    return NextResponse.json({ 
      status: playbook.vector_status,
      lastVectorized: playbook.last_vectorized,
      version: playbook.version,
      updatedAt: playbook.last_updated_at
    });

  } catch (error: any) {
    console.error("Status Check Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
