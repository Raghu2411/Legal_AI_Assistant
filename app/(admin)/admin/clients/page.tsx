import { getFirmClients } from "@/lib/clients/actions"
import { redirect } from "next/navigation"
import { createClient } from "@/lib/supabase/server"
import { ClientTable } from "@/components/clients/client-table"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { ShieldCheck } from "lucide-react"

export default async function AdminClientsPage({
  searchParams,
}: {
  searchParams: { q?: string }
}) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  // Check if user is admin
  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", user.id)
    .single()

  if (profile?.role !== "admin") {
    redirect("/access-denied")
  }

  const clients = await getFirmClients(searchParams.q)

  return (
    <div className="p-8 flex flex-col gap-8">
      <div className="flex items-center gap-3">
        <ShieldCheck className="h-8 w-8 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">Firm-Wide Oversight</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All Clients</CardTitle>
          <CardDescription>
            Admin Quality Control: Monitor all client portfolios across the entire firm.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ClientTable initialClients={clients} isAdmin={true} />
        </CardContent>
      </Card>
    </div>
  )
}
