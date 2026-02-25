import { createClient } from "@/lib/supabase/server"
import { redirect } from "next/navigation"
import { Button } from "@/components/ui/button"
import { signOut } from "@/app/auth/actions"

export default async function LawyerDashboard() {
  const supabase = createClient()
  const { data: { user } } = await supabase.auth.getUser()

  if (!user) {
    redirect("/login")
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-3xl font-bold">Lawyer Dashboard</h1>
        <form action={signOut}>
          <Button variant="outline">Sign Out</Button>
        </form>
      </div>
      <p>Welcome, {user.email}. You have access to your client portfolio.</p>
    </div>
  )
}
