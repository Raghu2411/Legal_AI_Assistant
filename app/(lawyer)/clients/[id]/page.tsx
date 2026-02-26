import { getClient } from "@/lib/clients/actions"
import { redirect, notFound } from "next/navigation"
import { createClient } from "@/lib/supabase/server"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { FileText, ArrowLeft, Calendar, Briefcase, User } from "lucide-react"
import Link from "next/link"

export default async function ClientOverviewPage({ params }: { params: { id: string } }) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  const client = await getClient(params.id)

  if (!client) {
    notFound()
  }

  return (
    <div className="p-8 flex flex-col gap-8 max-w-4xl mx-auto">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link href="/clients">
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <h1 className="text-3xl font-bold tracking-tight">Client Overview</h1>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="h-5 w-5 text-primary" />
              Identity
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Case ID</div>
              <div className="text-2xl font-mono font-bold">{client.auto_case_id}</div>
            </div>
            <div>
              <div className="text-sm font-medium text-muted-foreground">Full Name</div>
              <div className="text-xl font-semibold">{client.name}</div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Briefcase className="h-5 w-5 text-primary" />
              Case Details
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <div className="text-sm font-medium text-muted-foreground">Case Type</div>
              <div className="text-lg">{client.case_type}</div>
            </div>
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-medium text-muted-foreground">Status</div>
                <Badge className="mt-1" variant={client.status === 'Active' ? 'default' : 'secondary'}>
                  {client.status}
                </Badge>
              </div>
              <div className="text-right">
                <div className="text-sm font-medium text-muted-foreground">Onboarded</div>
                <div className="flex items-center gap-1 text-sm mt-1">
                  <Calendar className="h-3 w-3" />
                  {new Date(client.created_at).toLocaleDateString()}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Vault Access</CardTitle>
          <CardDescription>
            Securely manage all legal documents and evidence for this client.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button size="lg" className="w-full" asChild>
            <Link href={`/clients/${params.id}/vault`}>
              <FileText className="h-4 w-4 mr-2" />
              Open Document Vault
            </Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
