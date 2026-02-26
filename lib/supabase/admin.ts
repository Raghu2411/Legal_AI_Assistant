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

  // Helper to safely update if table exists
  const safeUpdate = async (table: string, column: string) => {
    try {
      const { error } = await supabase
        .from(table)
        .update({ [column]: adminId })
        .eq(column, lawyerId)
      
      if (error) {
        // If table doesn't exist, permission denied, or not in cache, we skip
        const isSkipable = 
          error.code === 'PGRST116' || 
          error.code === 'PGRST205' ||
          error.code === '42P01' || 
          error.code === '42501' ||
          error.status === 404 ||
          error.status === 403
        
        if (!isSkipable) {
          console.error(`Critical error reassigning ${table}:`, error)
          return false
        }
      }
      return true
    } catch (e) {
      console.warn(`Exception updating ${table}, skipping:`, e)
      return true
    }
  }

  const clientsOk = await safeUpdate('clients', 'assigned_to')
  const docsOk = await safeUpdate('documents', 'uploaded_by')

  return { success: clientsOk && docsOk }
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
