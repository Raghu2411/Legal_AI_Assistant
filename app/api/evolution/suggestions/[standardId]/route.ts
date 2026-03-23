import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";

export async function GET(
  request: Request,
  { params }: { params: { standardId: string } }
) {
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

  const { standardId } = params;

  try {
    const { data: suggestions, error } = await supabase
      .from("policy_suggestions")
      .select("*")
      .eq("standard_id", standardId)
      .eq("status", "pending")
      .order("created_at", { ascending: false });

    if (error) throw error;

    return NextResponse.json({ suggestions });

  } catch (error: any) {
    console.error("Fetch Suggestions Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
