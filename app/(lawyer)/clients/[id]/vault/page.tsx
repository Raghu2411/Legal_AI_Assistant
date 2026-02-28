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
import { VaultSearch } from "@/components/clients/vault-search"
import { retrieveContext } from "@/lib/ai/vector-service"
import { ArrowLeft, ShieldCheck, Search as SearchIcon } from "lucide-react"
import Link from "next/link"

export default async function ClientVaultPage({ 
  params,
  searchParams
}: { 
  params: { id: string },
  searchParams?: { q?: string }
}) {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  const client = await getClient(params.id)
  const documents = await getDocuments(params.id)

  if (!client) {
    notFound()
  }

  const query = searchParams?.q || ''
  let searchResults: any[] = []
  
  if (query) {
    try {
      searchResults = await retrieveContext(query, params.id)
    } catch (e) {
      console.error("Search failed", e)
    }
  }

  return (
    <div className="p-8 flex flex-col gap-8 max-w-5xl mx-auto">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link href={`/clients/${params.id}`}>
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Document Vault</h1>
            <p className="text-muted-foreground">Client: {client.name} ({client.auto_case_id})</p>
          </div>
        </div>
        <VaultSearch />
      </div>

      {query && (
        <Card className="border-primary/50 shadow-md">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SearchIcon className="h-5 w-5 text-primary" />
              Semantic Search Results
            </CardTitle>
            <CardDescription>
              Showing passages relevant to &quot;{query}&quot;
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {searchResults.length === 0 ? (
              <p className="text-sm text-muted-foreground italic">No relevant passages found.</p>
            ) : (
              searchResults.map((res, i) => (
                <div key={i} className="rounded-md border p-4 bg-muted/20">
                  <p className="text-sm">{res.content}</p>
                  <div className="mt-2 text-xs text-muted-foreground flex justify-between items-center">
                    <span>Similarity: {(res.similarity * 100).toFixed(1)}%</span>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      )}

      <div className="grid gap-8 md:grid-cols-[350px_1fr]">
        <div className="space-y-8">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <ShieldCheck className="h-5 w-5 text-primary" />
                Secure Upload
              </CardTitle>
              <CardDescription>
                Files are encrypted and isolated via Row-Level Security.
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

