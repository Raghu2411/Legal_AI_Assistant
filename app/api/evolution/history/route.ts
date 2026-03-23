import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";
import { getVersionHistory } from "@/lib/supabase/evolution-queries";

export async function GET(request: Request) {
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
    const history = await getVersionHistory();
    return NextResponse.json({ history });
  } catch (error: any) {
    console.error("History Fetch Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
