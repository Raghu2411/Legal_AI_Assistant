"use server";

import { createClient } from "@/lib/supabase/server";
import { revalidatePath } from "next/cache";

export async function confirmObligation(obligationId: string, dueDate?: string) {
  const supabase = createClient();
  
  const { data: ob } = await supabase
    .from("obligations")
    .select("client_id, description")
    .eq("id", obligationId)
    .single();

  const updateData: any = {
    status: 'confirmed',
    confirmed_at: new Date().toISOString(),
  };

  if (dueDate) {
    updateData.due_date = dueDate;
  }

  const { error } = await supabase
    .from("obligations")
    .update(updateData)
    .eq("id", obligationId);

  if (error) throw error;
  
  const { data: userData } = await supabase.auth.getUser();
  if (userData.user) {
    await supabase.from("activity_logs").insert({
      user_id: userData.user.id,
      client_id: ob?.client_id,
      action_type: 'OBLIGATION_CONFIRMED',
      metadata: { obligation_id: obligationId, description: ob?.description }
    });
  }

  revalidatePath("/dashboard/operations");
}

export async function rejectObligation(obligationId: string) {
  const supabase = createClient();

  const { data: ob } = await supabase
    .from("obligations")
    .select("client_id, description")
    .eq("id", obligationId)
    .single();

  const { error } = await supabase
    .from("obligations")
    .update({ status: 'rejected' })
    .eq("id", obligationId);

  if (error) throw error;
  
  const { data: userData } = await supabase.auth.getUser();
  if (userData.user) {
    await supabase.from("activity_logs").insert({
      user_id: userData.user.id,
      client_id: ob?.client_id,
      action_type: 'OBLIGATION_REJECTED',
      metadata: { obligation_id: obligationId, description: ob?.description }
    });
  }

  revalidatePath("/dashboard/operations");
}

export async function deleteObligation(obligationId: string) {
  const supabase = createClient();

  const { data: ob } = await supabase
    .from("obligations")
    .select("client_id, description")
    .eq("id", obligationId)
    .single();

  const { error } = await supabase
    .from("obligations")
    .delete()
    .eq("id", obligationId);

  if (error) throw error;

  const { data: userData } = await supabase.auth.getUser();
  if (userData.user) {
    await supabase.from("activity_logs").insert({
      user_id: userData.user.id,
      client_id: ob?.client_id,
      action_type: 'OBLIGATION_DELETED',
      metadata: { obligation_id: obligationId, description: ob?.description }
    });
  }

  revalidatePath("/dashboard/operations");
}

export async function revertObligationToPending(obligationId: string) {
  const supabase = createClient();

  const { data: ob } = await supabase
    .from("obligations")
    .select("client_id, description")
    .eq("id", obligationId)
    .single();

  const { error } = await supabase
    .from("obligations")
    .update({ 
      status: 'pending',
      confirmed_at: null 
    })
    .eq("id", obligationId);

  if (error) throw error;

  const { data: userData } = await supabase.auth.getUser();
  if (userData.user) {
    await supabase.from("activity_logs").insert({
      user_id: userData.user.id,
      client_id: ob?.client_id,
      action_type: 'OBLIGATION_REVERTED',
      metadata: { obligation_id: obligationId, description: ob?.description }
    });
  }

  revalidatePath("/dashboard/operations");
}
