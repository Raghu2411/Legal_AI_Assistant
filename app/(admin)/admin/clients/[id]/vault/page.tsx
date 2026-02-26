import { getClient, getDocuments } from "@/lib/clients/actions"
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
import { UploadForm } from "@/components/clients/upload-form"
import { VaultView } from "@/components/clients/vault-view"
import { ArrowLeft, ShieldCheck } from "lucide-react"
import Link from "next/link"

export default async function AdminClientVaultPage({ params }: { params: { id: string } }) {
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

  const client = await getClient(params.id)
  const documents = await getDocuments(params.id)

  if (!client) {
    notFound()
  }

  return (
    <div className="p-8 flex flex-col gap-8 max-w-5xl mx-auto">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="icon" asChild>
          <Link href={`/admin/clients/${params.id}`}>
            <ArrowLeft className="h-4 w-4" />
          </Link>
        </Button>
        <div>
          <div className="flex items-center gap-2">
            <ShieldCheck className="h-6 w-6 text-primary" />
            <h1 className="text-3xl font-bold tracking-tight">Admin Oversight: Document Vault</h1>
          </div>
          <p className="text-muted-foreground">Oversight for Client: {client.name} ({client.auto_case_id})</p>
        </div>
      </div>

      <div className="grid gap-8 md:grid-cols-[350px_1fr]">
        <div className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <ShieldCheck className="h-5 w-5 text-primary" />
                Secure Upload (Admin)
              </CardTitle>
              <CardDescription>
                Admins can upload documents to any client vault for quality control.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <UploadForm clientId={params.id} />
            </CardContent>
          </Card>
        </div>

        <div className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle>Vault Inventory</CardTitle>
              <CardDescription>
                All documents associated with this case file.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <VaultView clientId={params.id} documents={documents} />
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
