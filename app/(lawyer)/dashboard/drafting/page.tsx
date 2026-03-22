import { createClient } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import DraftingDashboard from "@/components/drafting/drafting-dashboard";

export default async function DraftingPage() {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const { data: clients, error } = await supabase
    .from("clients")
    .select("id, name, auto_case_id, case_type")
    .eq("lawyer_id", user.id)
    .order("name", { ascending: true });

  if (error) {
    return (
      <div className="p-8 text-destructive">
        Error loading clients: {error.message}
      </div>
    );
  }

  return <DraftingDashboard initialClients={clients || []} user={user} />;
}
