import { createClient } from "@/lib/supabase/server"
import { createAdminClient } from "@/lib/supabase/admin"
import { UserTable } from "@/components/admin/user-table"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { Users } from "lucide-react"

export default async function UserOversightPage() {
  const supabase = createClient()
  const adminSupabase = createAdminClient()

  const { data: profiles, error } = await adminSupabase
    .from("profiles")
    .select("*")
    .order("created_at", { ascending: false })

  if (error) {
    return (
      <div className="p-8 text-destructive">
        Error loading users: {error.message}
      </div>
    )
  }

  return (
    <div className="p-8 flex flex-col gap-8">
      <div className="flex items-center gap-3">
        <Users className="h-8 w-8 text-primary" />
        <h1 className="text-3xl font-bold tracking-tight">User Oversight</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>System Users</CardTitle>
          <CardDescription>
            Manage user roles and system access. Changes take effect immediately.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <UserTable initialUsers={profiles || []} />
        </CardContent>
      </Card>
    </div>
  )
}
