import { createClient } from "@/lib/supabase/server";
import { NextResponse } from "next/server";
import { logVersionHistory } from "@/lib/supabase/evolution-queries";

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
    const { historyId } = await request.json();

    // 1. Get the history entry to rollback to
    const { data: historyEntry, error: hError } = await supabase
      .from("version_history")
      .select("*")
      .eq("id", historyId)
      .single();

    if (hError || !historyEntry) throw new Error("History entry not found");

    // 2. Perform the rollback
    const table = historyEntry.entity_type === 'golden_rule' ? 'golden_rules' : 'playbooks';
    
    // Get current state for the "Before" of the rollback log
    const { data: current } = await supabase
      .from(table)
      .select("*")
      .eq("id", historyEntry.entity_id)
      .single();

    // Revert to old_value
    const updateData: any = {
      version: (current.version || 0) + 1,
      last_updated_by: user.id,
      last_updated_at: new Date().toISOString(),
    };

    if (historyEntry.entity_type === 'golden_rule') {
        updateData.admin_id = user.id;
        updateData.rule_text = historyEntry.old_value ? historyEntry.old_value.rule_text : "";
        // If old_value is null, it might mean we are rolling back a creation.
        // For the prototype, we'll revert to empty text or could consider deletion.
        // But usually history entries for updates have non-null old_values.
    } else {
        updateData.content = historyEntry.old_value;
    }

    const { error: uError } = await supabase
      .from(table)
      .update(updateData)
      .eq("id", historyEntry.entity_id);

    if (uError) throw uError;

    // 2.5 Reset Suggestion status if linked
    if (historyEntry.suggestion_id) {
        await supabase
          .from("policy_suggestions")
          .update({ status: 'pending' })
          .eq("id", historyEntry.suggestion_id);
    }

    // 3. Log the rollback action itself
    await logVersionHistory({
      entityType: historyEntry.entity_type,
      entityId: historyEntry.entity_id,
      field: historyEntry.field,
      oldValue: historyEntry.entity_type === 'golden_rule' ? { rule_text: current.rule_text } : current.content,
      newValue: historyEntry.old_value,
      changeType: 'rollback',
      userId: user.id,
    });

    return NextResponse.json({ message: "Rollback successful" });

  } catch (error: any) {
    console.error("Rollback Error:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}
