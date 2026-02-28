"use server"

import { createClient } from "@/lib/supabase/server"
import { createAdminClient, logEvent, reassignLawyerData } from "@/lib/supabase/admin"
import { revalidatePath } from "next/cache"

/**
 * Toggles a user's role between 'admin' and 'lawyer'.
 */
export async function toggleUserRole(userId: string, currentRole: string) {
  const supabase = createClient()
  const adminSupabase = createAdminClient()
  const { data: { user: adminUser } } = await supabase.auth.getUser()

  if (!adminUser) throw new Error("Unauthorized")

  // Check if current user is indeed an admin
  const { data: adminProfile } = await adminSupabase
    .from("profiles")
    .select("role")
    .eq("id", adminUser.id)
    .single()

  if (adminProfile?.role !== "admin") {
    throw new Error("Forbidden: Only admins can toggle roles")
  }

  const newRole = currentRole === "admin" ? "lawyer" : "admin"

  const { error } = await adminSupabase
    .from("profiles")
    .update({ role: newRole })
    .eq("id", userId)

  if (error) {
    console.error("Error updating role:", error)
    return { error: error.message }
  }

  await logEvent(adminUser.id, "ROLE_UPDATE", `Changed user ${userId} role from ${currentRole} to ${newRole}`)
  
  revalidatePath("/admin/users")
  return { success: true }
}

/**
 * Deletes a lawyer and reassigns their data to an admin.
 * Satisfies Constitution Principle VII.
 */
export async function deleteUser(userId: string) {
  const supabase = createClient()
  const adminSupabase = createAdminClient()
  const { data: { user: adminUser } } = await supabase.auth.getUser()

  if (!adminUser) throw new Error("Unauthorized")

  // 1. Mandatory data reassignment
  const { success: reassigned } = await reassignLawyerData(userId, adminUser.id)
  
  if (!reassigned) {
    return { error: "Failed to reassign lawyer data. Deletion aborted." }
  }

  // 2. Delete the profile
  const { error: profileError } = await adminSupabase
    .from("profiles")
    .delete()
    .eq("id", userId)

  if (profileError) {
    return { error: profileError.message }
  }

  // 3. Delete the auth user (requires admin client)
  const { error: authError } = await adminSupabase.auth.admin.deleteUser(userId)

  if (authError) {
    return { error: authError.message }
  }

  await logEvent(adminUser.id, "USER_DELETE", `Deleted user ${userId} and reassigned data to ${adminUser.id}`)

  revalidatePath("/admin/users")
  return { success: true }
}
