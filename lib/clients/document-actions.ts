"use server"

import { createClient } from "@/lib/supabase/server"
import { revalidatePath } from "next/cache"
import { extractTextFromFile, normalizeContext } from "@/lib/playbook/parser"
import { processDocument } from "@/lib/ai/vector-service"

export async function uploadDocumentAction(
  clientId: string, 
  formData: FormData
) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) return { error: "Not authenticated" }

  const file = formData.get("file") as File
  const docType = formData.get("docType") as string

  if (!file) return { error: "No file provided" }
  if (!docType) return { error: "No document type provided" }

  // Validate file type (Constitution Principle VIII)
  const allowedExtensions = ["pdf", "docx", "txt"]
  const fileExtension = file.name.split(".").pop()?.toLowerCase()
  if (!fileExtension || !allowedExtensions.includes(fileExtension)) {
    return { error: "Invalid file type. Only PDF, DOCX, and TXT are allowed." }
  }

  // 1. Extract text first (fail fast if unreadable)
  let normalizedText = ""
  try {
    const buffer = Buffer.from(await file.arrayBuffer())
    const extractedText = await extractTextFromFile(buffer, file.type || fileExtension)
    normalizedText = normalizeContext(extractedText)
  } catch (e: any) {
    return { error: `Failed to read document: ${e.message}` }
  }

  const filePath = `${clientId}/${crypto.randomUUID()}_${file.name}`

  // 2. Upload to Supabase Storage
  const { error: uploadError } = await supabase.storage
    .from("client-vaults")
    .upload(filePath, file)

  if (uploadError) return { error: uploadError.message }

  // 3. Insert record into documents table
  const { data, error: dbError } = await supabase
    .from("documents")
    .insert({
      client_id: clientId,
      file_url: filePath,
      file_name: file.name,
      doc_type: docType,
      uploaded_by: user.id,
    })
    .select()
    .single()

  if (dbError) {
    // Cleanup storage if DB insert fails
    await supabase.storage.from("client-vaults").remove([filePath])
    return { error: dbError.message }
  }

  // 4. Trigger vectorization asynchronously
  processDocument(data.id, clientId, normalizedText).catch(e => {
    console.error("Document vectorization failed:", e)
  })

  // Audit logging
  await supabase.from("logs").insert({
    user_id: user.id,
    event_type: "DOC_UPLOAD",
    description: `Uploaded ${docType}: ${file.name} to client vault`,
  })

  revalidatePath(`/(lawyer)/clients/${clientId}/vault`, "page")

  return { success: true, document: data }
}

export async function deleteDocumentAction(documentId: string, clientId: string, fileUrl: string) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) return { error: "Not authenticated" }

  // 1. Delete from storage
  const { error: storageError } = await supabase.storage
    .from("client-vaults")
    .remove([fileUrl])

  if (storageError) return { error: storageError.message }

  // 2. Delete from DB
  const { error: dbError } = await supabase
    .from("documents")
    .delete()
    .eq("id", documentId)

  if (dbError) return { error: dbError.message }

  // Audit logging
  await supabase.from("logs").insert({
    user_id: user.id,
    event_type: "DOC_DELETE",
    description: `Deleted document (ID: ${documentId}) from client vault`,
  })

  revalidatePath(`/(lawyer)/clients/${clientId}/vault`, "page")

  return { success: true }
}
