import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { signOut } from "@/app/auth/actions"
import { 
  Users, 
  FileText, 
  AlertCircle, 
  History, 
  Calendar,
  ArrowRight,
  ShieldCheck,
  Search,
  Eye,
  Activity
} from "lucide-react"
import Link from "next/link"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"

export default async function LawyerDashboard() {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  // 1. Fetch Stats
  const { count: clientCount } = await supabase
    .from("clients")
    .select("*", { count: 'exact', head: true })
    .eq("lawyer_id", user.id)

  const { count: docCount } = await supabase
    .from("documents")
    .select("*, clients!inner(lawyer_id)", { count: 'exact', head: true })
    .eq("clients.lawyer_id", user.id)

  const { count: pendingTriageCount } = await supabase
    .from("documents")
    .select("*, clients!inner(lawyer_id)", { count: 'exact', head: true })
    .eq("clients.lawyer_id", user.id)
    .eq("complexity_score", 0)

  const { count: pendingObligationCount } = await supabase
    .from("obligations")
    .select("*, clients!inner(lawyer_id)", { count: 'exact', head: true })
    .eq("clients.lawyer_id", user.id)
    .eq("status", "pending")

  // 2. Fetch Recent Activity (Combining 'logs' and 'activity_logs')
  const [logsResult, activityLogsResult] = await Promise.all([
    supabase
      .from("logs")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(5),
    supabase
      .from("activity_logs")
      .select("*")
      .eq("user_id", user.id)
      .order("created_at", { ascending: false })
      .limit(5)
  ])

  // Merge and sort logs
  const combinedLogs = [
    ...(logsResult.data || []).map(l => ({ ...l, type: 'system' })),
    ...(activityLogsResult.data || []).map(l => ({ ...l, type: 'activity', event_type: l.action_type }))
  ]
  .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
  .slice(0, 5)

  // 3. Fetch Recent Clients (Portfolio View)
  const { data: recentClients } = await supabase
    .from("clients")
    .select("id, name, case_type, auto_case_id, status, created_at")
    .eq("lawyer_id", user.id)
    .order("created_at", { ascending: false })
    .limit(5)

  return (
    <div className="p-4 md:p-8 flex flex-col gap-8 max-w-7xl mx-auto w-full">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Lawyer Dashboard</h1>
          <p className="text-muted-foreground">Welcome back. Here is an overview of your practice.</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" asChild>
            <Link href="/clients">View All Clients</Link>
          </Button>
          <form action={signOut}>
            <Button variant="ghost">Sign Out</Button>
          </form>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Clients</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{clientCount || 0}</div>
            <p className="text-xs text-muted-foreground">Portfolio size</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Documents</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{docCount || 0}</div>
            <p className="text-xs text-muted-foreground">Managed in vault</p>
          </CardContent>
        </Card>
        <Card className={pendingTriageCount && pendingTriageCount > 0 ? "border-amber-500/50 bg-amber-500/5" : ""}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Triage</CardTitle>
            <AlertCircle className={`h-4 w-4 ${pendingTriageCount && pendingTriageCount > 0 ? "text-amber-500" : "text-muted-foreground"}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingTriageCount || 0}</div>
            <p className="text-xs text-muted-foreground">Awaiting classification</p>
          </CardContent>
        </Card>
        <Card className={pendingObligationCount && pendingObligationCount > 0 ? "border-primary/50 bg-primary/5" : ""}>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Obligations</CardTitle>
            <Calendar className={`h-4 w-4 ${pendingObligationCount && pendingObligationCount > 0 ? "text-primary" : "text-muted-foreground"}`} />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{pendingObligationCount || 0}</div>
            <p className="text-xs text-muted-foreground">Awaiting confirmation</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-8 lg:grid-cols-7">
        {/* Left Column: Client Portfolio */}
        <Card className="lg:col-span-4">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5 text-primary" />
                  Recent Clients
                </CardTitle>
                <CardDescription>
                  Your most recently added clients and their case status.
                </CardDescription>
              </div>
              <Button variant="ghost" size="sm" asChild>
                <Link href="/clients" className="text-xs">
                  View All <ArrowRight className="h-3 w-3 ml-1" />
                </Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {recentClients && recentClients.length > 0 ? (
              <div className="rounded-md border overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Client Name</TableHead>
                      <TableHead>Case Type</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Action</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {recentClients.map((client) => (
                      <TableRow key={client.id}>
                        <TableCell className="font-medium">
                          <div>
                            <p>{client.name}</p>
                            <p className="text-[10px] text-muted-foreground font-mono">{client.auto_case_id}</p>
                          </div>
                        </TableCell>
                        <TableCell className="text-sm">{client.case_type}</TableCell>
                        <TableCell>
                          <Badge variant={client.status === 'Active' ? 'default' : 'secondary'} className="text-[10px] h-5">
                            {client.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button variant="ghost" size="icon" asChild className="h-8 w-8">
                            <Link href={`/clients/${client.id}/vault`}>
                              <FileText className="h-4 w-4" />
                            </Link>
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-center">
                <div className="rounded-full bg-primary/10 p-3 mb-4">
                  <Users className="h-6 w-6 text-primary" />
                </div>
                <h3 className="font-semibold">No clients yet</h3>
                <p className="text-sm text-muted-foreground max-w-[250px]">
                  Start by adding your first client to the portfolio.
                </p>
                <Button className="mt-4" size="sm" asChild>
                  <Link href="/clients">Add Client</Link>
                </Button>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right Column: Activity Feed */}
        <Card className="lg:col-span-3">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <History className="h-5 w-5 text-primary" />
              Recent Activity
            </CardTitle>
            <CardDescription>Your latest actions in the system.</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {combinedLogs && combinedLogs.length > 0 ? (
                combinedLogs.map((log) => (
                  <div key={log.id} className="flex items-start gap-4">
                    <div className="mt-1 bg-muted rounded-full p-2 shrink-0">
                      {log.type === 'system' ? (
                        <ShieldCheck className="h-3 w-3 text-muted-foreground" />
                      ) : (
                        <Activity className="h-3 w-3 text-primary" />
                      )}
                    </div>
                    <div className="flex-1 space-y-1 overflow-hidden">
                      <p className="text-sm font-medium leading-none truncate">
                        {log.event_type.replace(/_/g, ' ')}
                      </p>
                      <p className="text-xs text-muted-foreground line-clamp-2">
                        {log.description || (log.metadata?.description) || "No description provided."}
                      </p>
                      <p className="text-[10px] text-muted-foreground font-mono">
                        {new Date(log.created_at).toLocaleString()}
                      </p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground py-4 text-center">No recent activity found.</p>
              )}
            </div>
            {combinedLogs && combinedLogs.length > 0 && (
              <Button variant="ghost" className="w-full mt-6 text-xs gap-2" size="sm" asChild>
                <Link href="/clients">
                  View Full Portfolio <ArrowRight className="h-3 w-3" />
                </Link>
              </Button>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
