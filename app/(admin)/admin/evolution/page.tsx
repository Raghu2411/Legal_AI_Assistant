import { createClient } from "@/lib/supabase/server";
import { redirect } from "next/navigation";
import { EvolutionStudioDashboard } from "@/components/evolution/evolution-studio-dashboard";

export default async function EvolutionPage() {
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    redirect("/login");
  }

  const { data: profile } = await supabase
    .from("profiles")
    .select("role")
    .eq("id", user.id)
    .single();

  if (profile?.role !== 'admin') {
    redirect("/access-denied");
  }

  // Fetch initial data
  const { data: standards } = await supabase
    .from("compliance_standards")
    .select("*")
    .order("uploaded_at", { ascending: false });

  const { data: playbooks } = await supabase
    .from("playbooks")
    .select("*")
    .order("version", { ascending: false });

  const { data: goldenRules } = await supabase
    .from("golden_rules")
    .select("*")
    .order("priority", { ascending: true });

  return (
    <div className="container mx-auto py-10 space-y-8">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Evolution Studio</h1>
        <p className="text-muted-foreground">
          Audit firm legal logic against external standards and evolve your policies.
        </p>
      </div>

      <EvolutionStudioDashboard 
        initialStandards={standards || []}
        playbooks={playbooks || []}
        goldenRules={goldenRules || []}
      />
    </div>
  );
}
