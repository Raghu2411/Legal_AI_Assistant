"use server"

import { createClient } from "@/lib/supabase/server"
import { revalidatePath } from "next/cache"
import { clientSchema } from "./schemas"

// T007: Implement createClient server action
export async function createClientAction(formData: FormData) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    return { error: "Not authenticated" }
  }

  const name = formData.get("name") as string
  const case_type = formData.get("case_type") as string

  const validated = clientSchema.safeParse({ name, case_type })

  if (!validated.success) {
    return { error: validated.error.errors[0].message }
  }

  const { data, error } = await supabase
    .from("clients")
    .insert({
      name: validated.data.name,
      case_type: validated.data.case_type,
      lawyer_id: user.id,
      status: "Active",
    })
    .select()
    .single()

  if (error) {
    return { error: error.message }
  }

  // T021: Integrate audit logging (early integration)
  await supabase.from("logs").insert({
    user_id: user.id,
    event_type: "CLIENT_CREATE",
    description: `Added client: ${validated.data.name} (${validated.data.case_type})`,
  })

  revalidatePath("/(lawyer)/clients", "page")
  revalidatePath("/(admin)/admin/clients", "page")

  return { success: true, client: data }
}

export async function getClient(clientId: string) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) return null

  const { data, error } = await supabase
    .from("clients")
    .select("*")
    .eq("id", clientId)
    .single()

  if (error) {
    console.error("Error fetching client:", error)
    return null
  }

  return data
}

export async function getDocuments(clientId: string) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) return []

  const { data, error } = await supabase
    .from("documents")
    .select("*")
    .eq("client_id", clientId)
    .order("uploaded_at", { ascending: false })

  if (error) {
    console.error("Error fetching documents:", error)
    return []
  }

  return data || []
}

// T017: Implement getFirmClients and updateClient server actions
export async function getFirmClients(searchQuery?: string) {
  const supabase = createClient()
  
  let query = supabase
    .from("clients")
    .select(`
      *,
      profiles:lawyer_id (
        full_name
      )
    `)
    .order("created_at", { ascending: false })

  if (searchQuery) {
    // Search by client name or lawyer name
    // Since we can't easily filter by joined table in a single ilike without complex setup, 
    // we'll filter by client name first. 
    // For lawyer name search, we might need a more advanced query or filter in-memory if results are small.
    query = query.ilike("name", `%${searchQuery}%`)
  }

  const { data, error } = await query

  if (error) {
    console.error("Error fetching firm clients:", error)
    return []
  }

  return data || []
}

export async function updateClientAction(clientId: string, updates: any) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) return { error: "Not authenticated" }

  const { data, error } = await supabase
    .from("clients")
    .update(updates)
    .eq("id", clientId)
    .select()
    .single()

  if (error) return { error: error.message }

  // Audit logging
  await supabase.from("logs").insert({
    user_id: user.id,
    event_type: "CLIENT_UPDATE",
    description: `Updated client: ${data.name} (ID: ${data.auto_case_id})`,
  })

  revalidatePath("/(lawyer)/clients", "page")
  revalidatePath("/(admin)/admin/clients", "page")
  revalidatePath(`/(lawyer)/clients/${clientId}`, "page")

  return { success: true, client: data }
}

// T012: Implement uploadDocument and deleteDocument server actions
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

  const filePath = `${clientId}/${crypto.randomUUID()}_${file.name}`

  // 1. Upload to Supabase Storage
  const { error: uploadError } = await supabase.storage
    .from("client-vaults")
    .upload(filePath, file)

  if (uploadError) return { error: uploadError.message }

  // 2. Insert record into documents table
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
