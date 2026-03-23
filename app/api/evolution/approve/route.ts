import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";
import { approveSuggestion } from "@/lib/supabase/evolution-queries";

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
    const { suggestionIds, expectedVersions } = await request.json();

    if (!suggestionIds || !Array.isArray(suggestionIds)) {
      return NextResponse.json({ error: "Invalid suggestion IDs" }, { status: 400 });
    }

    // Process each approval with concurrency check (US2/T026)
    for (const id of suggestionIds) {
      const { data: suggestion } = await supabase
        .from("policy_suggestions")
        .select("target_id, target_type")
        .eq("id", id)
        .single();

      const isNewPlaceholder = suggestion?.target_id === '00000000-0000-0000-0000-000000000000';

      if (suggestion && expectedVersions && expectedVersions[suggestion.target_id] && !isNewPlaceholder) {
        const table = suggestion.target_type === 'golden_rule' ? 'golden_rules' : 'playbooks';
        const { data: current } = await supabase
          .from(table)
          .select("version")
          .eq("id", suggestion.target_id)
          .single();

        if (current && current.version !== expectedVersions[suggestion.target_id]) {
          return NextResponse.json({ 
            error: `Concurrency Error: The rule/clause for ${suggestion.target_id} has been updated by another admin. Please refresh and review again.`,
            id: suggestion.target_id
          }, { status: 409 });
        }
      }
      
      await approveSuggestion(id, user.id);
    }

    return NextResponse.json({ 
        message: `Successfully approved ${suggestionIds.length} suggestions.` 
    });

  } catch (error: any) {
    console.error("Approval Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
