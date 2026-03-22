import { createClient } from "@/lib/supabase/server";
import { TriageTable } from "@/components/operations/triage-table";
import { VerificationList } from "@/components/operations/verification-list";
import { CalendarView } from "@/components/operations/calendar-view";
import { ClientFilter } from "@/components/operations/client-filter";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default async function OperationsPage({
  searchParams,
}: {
  searchParams: { clientId?: string };
}) {
  const supabase = createClient();
  const clientId = searchParams.clientId;

  // Fetch clients for the filter
  const { data: clients } = await supabase
    .from("clients")
    .select("id, name")
    .order("name");

  // Fetch documents for Triage Queue
  let query = supabase
    .from("documents")
    .select("id, file_name, classification, complexity_score, uploaded_at, client_id")
    .order("uploaded_at", { ascending: false });

  if (clientId) {
    query = query.eq("client_id", clientId);
  }

  const { data: documents } = await query;

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Operations Dashboard</h1>
        <ClientFilter clients={clients || []} />
      </div>

      <Tabs defaultValue="triage" className="w-full">
        <TabsList>
          <TabsTrigger value="triage">Triage Queue</TabsTrigger>
          <TabsTrigger value="calendar">Operations Calendar</TabsTrigger>
          <TabsTrigger value="obligations">Verification</TabsTrigger>
        </TabsList>
        <TabsContent value="triage" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Triage Queue</CardTitle>
              <CardDescription>
                AI-classified documents based on firm Golden Rules.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <TriageTable initialDocuments={documents || []} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="calendar">
          <Card>
            <CardHeader>
              <CardTitle>Operations Calendar</CardTitle>
              <CardDescription>
                Confirmed milestones and legal obligations.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <CalendarView clientId={clientId} />
            </CardContent>
          </Card>
        </TabsContent>
        <TabsContent value="obligations">
          <Card>
            <CardHeader>
              <CardTitle>Obligation Verification</CardTitle>
              <CardDescription>
                Review and confirm AI-extracted obligations.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <VerificationList clientId={clientId} />
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
