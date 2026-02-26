import { createClient } from "@/lib/supabase/server"
import { PlaybookForm } from "@/components/admin/playbook-form"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { BookOpen } from "lucide-react"
import { LogTable } from "@/components/admin/log-table"

export default async function PlaybookPage() {
  const supabase = createClient()

  // 1. Fetch latest playbook/rules
  const { data: latestPlaybook } = await supabase
    .from("playbooks")
    .select("*")
    .order("version", { ascending: false })
    .limit(1)
    .single()

  // 2. Fetch playbook upload history from logs
  const { data: rawHistory } = await supabase
    .from("logs")
    .select("*")
    .eq("event_type", "PLAYBOOK_UPLOAD")
    .order("created_at", { ascending: false })
    .limit(10)

  // Manual join profiles for history
  const { data: profiles } = await supabase
    .from("profiles")
    .select("id, full_name")

  const profileMap = new Map(profiles?.map(p => [p.id, p.full_name]) || [])
  const history = rawHistory?.map(log => ({
    ...log,
    profiles: {
      full_name: profileMap.get(log.user_id) || "System"
    }
  })) || []

  return (
    <div className="p-8 flex flex-col gap-8">
      <div className="flex items-center gap-3">
        <BookOpen className="h-8 w-8 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">Playbook & Rules</h1>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="space-y-8">
          <PlaybookForm 
            initialRules={latestPlaybook?.golden_rules || ""} 
            currentVersion={latestPlaybook?.version || 0} 
          />
        </div>

        <div className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Upload History</CardTitle>
              <CardDescription>
                Recent changes to the firm's Legal Playbook.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <LogTable logs={history} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Active Configuration</CardTitle>
              <CardDescription>
                Details of the currently active AI context source.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-sm">
              <div className="flex justify-between border-b pb-2">
                <span className="text-muted-foreground">Version</span>
                <span className="font-medium">v{latestPlaybook?.version || "N/A"}</span>
              </div>
              <div className="flex justify-between border-b pb-2">
                <span className="text-muted-foreground">Last Updated</span>
                <span className="font-medium">
                  {latestPlaybook?.created_at 
                    ? new Date(latestPlaybook.created_at).toLocaleString() 
                    : "N/A"}
                </span>
              </div>
              <div className="flex justify-between border-b pb-2">
                <span className="text-muted-foreground">Current File</span>
                <span className="font-medium">{latestPlaybook?.file_name || "None"}</span>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
