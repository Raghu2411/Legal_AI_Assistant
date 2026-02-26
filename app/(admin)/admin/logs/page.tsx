import { createClient } from "@/lib/supabase/server"
import { LogTable } from "@/components/admin/log-table"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { History } from "lucide-react"

export default async function AuditTrailPage() {
  const supabase = createClient()

  // 1. Fetch all logs
  const { data: rawLogs, error: logError } = await supabase
    .from("logs")
    .select("*")
    .order("created_at", { ascending: false })
    .limit(100)

  console.log('AuditTrailPage: rawLogs count:', rawLogs?.length);

  if (logError) {
    console.error('AuditTrailPage: logError:', logError);
    return (
      <div className="p-8 text-destructive">
        Error loading logs: {logError.message}
      </div>
    )
  }

  // 2. Fetch all profiles for joining manually
  const { data: profiles, error: profileError } = await supabase
    .from("profiles")
    .select("id, full_name")
  
  console.log('AuditTrailPage: profiles count:', profiles?.length);
  if (profileError) console.error('AuditTrailPage: profileError:', profileError);

  // 3. Map names to logs
  const profileMap = new Map(profiles?.map(p => [p.id, p.full_name]) || [])
  const logs = rawLogs.map(log => ({
    ...log,
    profiles: {
      full_name: profileMap.get(log.user_id) || "System"
    }
  }))

  return (
    <div className="p-8 flex flex-col gap-8">
      <div className="flex items-center gap-3">
        <History className="h-8 w-8 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">Audit Trail</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Activity</CardTitle>
          <CardDescription>
            A chronological history of actions performed within the system.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <LogTable logs={logs as any || []} />
        </CardContent>
      </Card>
    </div>
  )
}
