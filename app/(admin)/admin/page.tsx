import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { DashboardContent } from "@/components/admin/dashboard-content"

export default async function AdminDashboard() {
  const supabase = createClient()
  const { data } = await supabase.auth.getUser()
  const user = data?.user

  if (!user) {
    redirect("/login")
  }

  // Fetch initial summary stats
  const { count: userCount } = await supabase
    .from("profiles")
    .select("*", { count: "exact", head: true })

  // FR-011: Fetch logs from the last 24 hours
  const twentyFourHoursAgo = new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString()
  
  const { count: recentLogCount } = await supabase
    .from("logs")
    .select("*", { count: "exact", head: true })
    .gte("created_at", twentyFourHoursAgo)

  const { data: rawLogs } = await supabase
    .from("logs")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(5)

  const { data: profiles } = await supabase
    .from("profiles")
    .select("id, full_name")

  const profileMap = new Map(profiles?.map(p => [p.id, p.full_name]) || [])
  const latestLogs = rawLogs?.map(log => ({
    ...log,
    profiles: {
      full_name: profileMap.get(log.user_id) || "System"
    }
  })) || []

  const { data: latestPlaybook } = await supabase
    .from("playbooks")
    .select("version")
    .order("version", { ascending: false })
    .limit(1)
    .single()

  return (
    <DashboardContent 
      initialUserCount={userCount || 0}
      initialLogs={latestLogs || []}
      userEmail={user.email || ""}
      currentVersion={latestPlaybook?.version || 0}
      recentLogCount={recentLogCount || 0}
    />
  )
}
