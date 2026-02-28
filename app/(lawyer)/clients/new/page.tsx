import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { ClientForm } from "@/components/clients/client-form"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { UserPlus } from "lucide-react"

export default async function NewClientPage() {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <div className="flex items-center gap-3 mb-8">
        <UserPlus className="h-8 w-8 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">Onboard New Client</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Client Details</CardTitle>
          <CardDescription>
            Enter the client information to generate a unique firm-wide Case ID.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ClientForm />
        </CardContent>
      </Card>
    </div>
  )
}
