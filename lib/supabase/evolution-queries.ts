import { createClient } from "./server";
import { Database } from "@/types/supabase";

export async function logVersionHistory({
  entityType,
  entityId,
  field,
  oldValue,
  newValue,
  changeType = 'update',
  userId,
  suggestionId,
}: {
  entityType: 'playbook' | 'golden_rule';
  entityId: string;
  field?: string;
  oldValue: any;
  newValue: any;
  changeType?: string;
  userId: string;
  suggestionId?: string;
}) {
  const supabase = createClient();

  const { error } = await supabase
    .from("version_history")
    .insert({
      entity_type: entityType,
      entity_id: entityId,
      field,
      old_value: oldValue,
      new_value: newValue,
      change_type: changeType,
      user_id: userId,
      suggestion_id: suggestionId,
    });

  if (error) {
    console.error("Failed to log version history:", error);
    throw error;
  }
}

export async function getVersionHistory(entityType?: string, entityId?: string) {
  const supabase = createClient();

  let query = supabase
    .from("version_history")
    .select("*")
    .order("timestamp", { ascending: false });

  if (entityType) {
    query = query.eq("entity_type", entityType);
  }

  if (entityId) {
    query = query.eq("entity_id", entityId);
  }

  const { data, error } = await query;

  if (error) {
    console.error("Failed to fetch version history:", error);
    throw error;
  }

  return data;
}

export async function createGoldenRule(
  ruleText: string,
  userId: string,
  suggestionId?: string
) {
  const supabase = createClient();

  const { data: newRule, error: insertError } = await supabase
    .from("golden_rules")
    .insert({
      admin_id: userId,
      rule_text: ruleText,
      version: 1,
      last_updated_by: userId,
      last_updated_at: new Date().toISOString(),
    })
    .select()
    .single();

  if (insertError) throw insertError;

  // Log history
  await logVersionHistory({
    entityType: 'golden_rule',
    entityId: newRule.id,
    field: 'rule_text',
    oldValue: null,
    newValue: { rule_text: ruleText },
    changeType: 'create',
    userId,
    suggestionId,
  });

  return newRule;
}

export async function updateGoldenRule(
  id: string,
  ruleText: string,
  userId: string,
  suggestionId?: string
) {
  const supabase = createClient();

  // 1. Get current state for history
  const { data: current, error: fetchError } = await supabase
    .from("golden_rules")
    .select("*")
    .eq("id", id)
    .single();

  if (fetchError) throw fetchError;

  // 2. Perform atomic update
  const { data: updated, error: updateError } = await supabase
    .from("golden_rules")
    .update({
      admin_id: userId,
      rule_text: ruleText,
      version: (current.version || 0) + 1,
      last_updated_by: userId,
      last_updated_at: new Date().toISOString(),
    })
    .eq("id", id)
    .select()
    .single();

  if (updateError) throw updateError;

  // 3. Log history
  await logVersionHistory({
    entityType: 'golden_rule',
    entityId: id,
    field: 'rule_text',
    oldValue: { rule_text: current.rule_text },
    newValue: { rule_text: ruleText },
    userId,
    suggestionId,
  });

  return updated;
}

export async function approveSuggestion(
  suggestionId: string,
  userId: string
) {
  const supabase = createClient();

  // 1. Get suggestion details
  const { data: suggestion, error: fetchError } = await supabase
    .from("policy_suggestions")
    .select("*")
    .eq("id", suggestionId)
    .single();

  if (fetchError) throw fetchError;

  const isNew = suggestion.target_id === '00000000-0000-0000-0000-000000000000';

  // 2. Apply change based on target type
  if (suggestion.target_type === 'golden_rule') {
    if (isNew) {
      await createGoldenRule(suggestion.suggested_text, userId, suggestionId);
    } else {
      await updateGoldenRule(suggestion.target_id, suggestion.suggested_text, userId, suggestionId);
    }
  } else if (suggestion.target_type === 'playbook') {
    if (isNew) {
      // In a real system, you might create a new playbook section
      console.warn("New playbook section creation not implemented for prototype.");
    } else {
      const { data: playbook, error: pError } = await supabase
        .from("playbooks")
        .select("*")
        .eq("id", suggestion.target_id)
        .single();

      if (pError) throw pError;

      const { error: upError } = await supabase
        .from("playbooks")
        .update({
          content: suggestion.suggested_text as any,
          version: (playbook.version || 0) + 1,
          last_updated_by: userId,
          last_updated_at: new Date().toISOString(),
        })
        .eq("id", suggestion.target_id);

      if (upError) throw upError;

      await logVersionHistory({
        entityType: 'playbook',
        entityId: suggestion.target_id,
        oldValue: playbook.content,
        newValue: suggestion.suggested_text,
        userId,
        suggestionId,
      });
    }
  }

  // 3. Mark suggestion as approved
  const { error: sUpdateError } = await supabase
    .from("policy_suggestions")
    .update({ status: 'approved' })
    .eq("id", suggestionId);

  if (sUpdateError) throw sUpdateError;
}
