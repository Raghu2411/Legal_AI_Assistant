import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { ClientTable } from "@/components/clients/client-table"
import { Button } from "@/components/ui/button"
import { PlusCircle, Users } from "lucide-react"
import Link from "next/link"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"

export default async function LawyerClientsPage() {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  const { data: clients, error } = await supabase
    .from("clients")
    .select("*")
    .eq("lawyer_id", user.id)
    .order("created_at", { ascending: false })

  if (error) {
    return (
      <div className="p-8 text-destructive">
        Error loading clients: {error.message}
      </div>
    )
  }

  if (clients?.length === 0) {
    return (
      <div className="p-8 flex flex-col items-center justify-center min-h-[400px] border-2 border-dashed rounded-lg text-center gap-4">
        <Users className="h-12 w-12 text-muted-foreground" />
        <div>
          <h2 className="text-2xl font-semibold">No Clients Found</h2>
          <p className="text-muted-foreground">You haven&apos;t onboarded any clients yet. Start by adding your first client.</p>
        </div>
        <Button asChild>
          <Link href="/clients/new">
            <PlusCircle className="h-4 w-4 mr-2" />
            Add First Client
          </Link>
        </Button>
      </div>
    )
  }

  return (
    <div className="p-8 flex flex-col gap-8">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Users className="h-8 w-8 text-primary" />
          <h1 className="text-3xl font-bold tracking-tight">My Client Portfolio</h1>
        </div>
        <Button asChild>
          <Link href="/clients/new">
            <PlusCircle className="h-4 w-4 mr-2" />
            Add Client
          </Link>
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Clients</CardTitle>
          <CardDescription>
            Manage your assigned clients and access their document vaults.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ClientTable initialClients={clients || []} />
        </CardContent>
      </Card>
    </div>
  )
}
