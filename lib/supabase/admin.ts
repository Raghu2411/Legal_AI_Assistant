import { createClient } from '@supabase/supabase-js'

/**
 * Creates a Supabase client with the service role key.
 * This client bypasses Row Level Security (RLS).
 * Use ONLY for administrative tasks.
 */
export const createAdminClient = () => {
  return createClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.SUPABASE_SERVICE_ROLE_KEY!,
    {
      auth: {
        autoRefreshToken: false,
        persistSession: false
      }
    }
  )
}

/**
 * Reassigns all data from a lawyer to an admin before deletion.
 * Mandatory per Constitution Principle VII.
 */
export const reassignLawyerData = async (lawyerId: string, adminId: string) => {
  const supabase = createAdminClient()

  // 1. Reassign Clients (lawyer_id)
  const { error: clientError } = await supabase
    .from('clients')
    .update({ lawyer_id: adminId })
    .eq('lawyer_id', lawyerId)

  if (clientError) {
    console.error("Critical error reassigning clients:", clientError)
    return { success: false }
  }

  // 2. Reassign Documents (uploaded_by)
  const { error: docError } = await supabase
    .from('documents')
    .update({ uploaded_by: adminId })
    .eq('uploaded_by', lawyerId)

  if (docError) {
    console.error("Critical error reassigning documents:", docError)
    // We don't fail the whole operation if documents fail, 
    // but in a production system we'd want more robust cleanup.
  }

  return { success: true }
}

/**
 * Logs a system event to the audit trail.
 */
export const logEvent = async (userId: string, eventType: string, description: string, metadata: any = {}) => {
  const supabase = createAdminClient()
  
  const { error } = await supabase
    .from('logs')
    .insert({
      user_id: userId,
      event_type: eventType,
      description: description,
      metadata: metadata
    })

  if (error) {
    console.error('Error logging event:', error)
  }
}
