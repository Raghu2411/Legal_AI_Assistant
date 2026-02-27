"use server"

import { createClient } from "@/lib/supabase/server"
import { createAdminClient, logEvent } from "@/lib/supabase/admin"
import { extractTextFromFile, normalizeContext } from "@/lib/playbook/parser"
import { processDocument } from "@/lib/ai/vector-service"
import { revalidatePath } from "next/cache"

export async function uploadPlaybook(formData: FormData) {
  const supabase = createClient()
  const adminSupabase = createAdminClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) throw new Error("Unauthorized")

  const file = formData.get("file") as File
  const goldenRules = formData.get("golden_rules") as string

  if (!file) throw new Error("No file provided")

  // 1. Get current version
  const { data: latest } = await supabase
    .from("playbooks")
    .select("version")
    .order("version", { ascending: false })
    .limit(1)
    .single()

  const nextVersion = (latest?.version || 0) + 1
  const extension = file.name.split('.').pop() || 'pdf'
  const fileName = `playbook_v${nextVersion}.${extension}`
  const filePath = `versions/${fileName}`

  // 2. Extract text for AI context
  const buffer = Buffer.from(await file.arrayBuffer())
  const extractedText = await extractTextFromFile(buffer, file.type || extension)
  const normalizedText = normalizeContext(extractedText)
  
  console.log(`uploadPlaybook: Normalized text length: ${normalizedText.length}`);

  if (normalizedText.length === 0) {
    console.error("uploadPlaybook: No text could be extracted from the file.");
    // We still allow the upload, but we should probably warn or handle this state
  }

  // 3. Upload to Storage
  const { error: storageError } = await adminSupabase.storage
    .from("playbooks")
    .upload(filePath, file)

  if (storageError) {
    return { error: storageError.message }
  }

  // 4. Save metadata to DB
  const { data: insertedPlaybook, error: dbError } = await supabase
    .from("playbooks")
    .insert({
      file_path: filePath,
      file_name: file.name,
      golden_rules: goldenRules,
      version: nextVersion,
      created_by: user.id
    })
    .select("id")
    .single()

  if (dbError) {
    return { error: dbError.message }
  }

  // 5. Trigger vectorization asynchronously
  processDocument(insertedPlaybook.id, null, normalizedText, "playbooks").catch(e => {
    console.error("Vectorization failed:", e)
  })

  await logEvent(user.id, "PLAYBOOK_UPLOAD", `Uploaded playbook version ${nextVersion} (${file.name})`)

  revalidatePath("/admin/playbook")
  revalidatePath("/admin")
  return { success: true }
}

export async function updateGoldenRules(goldenRules: string) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) throw new Error("Unauthorized")

  // Update the latest playbook record's golden rules
  // In a real app, you might want to create a new version record even for text changes
  const { data: latest } = await supabase
    .from("playbooks")
    .select("id")
    .order("version", { ascending: false })
    .limit(1)
    .single()

  if (!latest) {
    // If no playbook exists, create a shell record
    const { error } = await supabase
      .from("playbooks")
      .insert({
        golden_rules: goldenRules,
        version: 1,
        created_by: user.id
      })
    if (error) return { error: error.message }
  } else {
    const { error } = await supabase
      .from("playbooks")
      .update({ golden_rules: goldenRules })
      .eq("id", latest.id)
    if (error) return { error: error.message }
  }

  await logEvent(user.id, "GOLDEN_RULES_UPDATE", "Updated firm-wide Golden Rules")

  revalidatePath("/admin/playbook")
  return { success: true }
}
