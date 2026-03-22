'use client';

import { Suspense, useState } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ArrowLeft, MessageSquare, FileText, ShieldCheck } from 'lucide-react';
import Link from 'next/link';
import { ChatPanel } from '@/components/intelligence-hub/chat-panel';
import { BriefingPanel } from '@/components/intelligence-hub/briefing-panel';
import { VendorToggle } from '@/components/intelligence-hub/vendor-toggle';

export default function IntelligenceHubPage({
  searchParams,
}: {
  searchParams: { clientId?: string };
}) {
  const [isVendorOnly, setIsVendorOnly] = useState(false);
  const clientId = searchParams.clientId;

  if (!clientId) {
    return (
      <div className="p-8 flex flex-col items-center justify-center min-h-[50vh] gap-4">
        <h1 className="text-2xl font-bold">No Client Selected</h1>
        <p className="text-muted-foreground text-center max-w-md">
          Please select a client from the dashboard to access the Intelligence Hub.
        </p>
        <Button asChild>
          <Link href="/clients">View Clients</Link>
        </Button>
      </div>
    );
  }

  return (
    <div className="p-8 flex flex-col gap-8 max-w-6xl mx-auto h-[calc(100vh-4rem)]">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" asChild>
            <Link href={`/clients/${clientId}`}>
              <ArrowLeft className="h-4 w-4" />
            </Link>
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">Intelligence Hub</h1>
        </div>
      </div>

      <Tabs defaultValue="chat" className="flex-1 flex flex-col gap-4 overflow-hidden">
        <TabsList className="bg-muted w-fit p-1">
          <TabsTrigger value="chat" className="flex items-center gap-2">
            <MessageSquare className="h-4 w-4" />
            Chat
          </TabsTrigger>
          <TabsTrigger value="briefing" className="flex items-center gap-2">
            <FileText className="h-4 w-4" />
            Briefing
          </TabsTrigger>
          <TabsTrigger value="vendor" className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4" />
            Vendor Mode
          </TabsTrigger>
        </TabsList>

        <div className="flex-1 overflow-hidden">
          <TabsContent value="chat" className="h-full m-0">
            <Card className="h-full flex flex-col">
              <CardHeader className="pb-3 flex flex-row items-center justify-between space-y-0">
                <div>
                  <CardTitle>Client Intelligence Chat</CardTitle>
                  <CardDescription>
                    Conversational AI grounded in this client's document vault.
                  </CardDescription>
                </div>
                {isVendorOnly && (
                  <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20">
                    <ShieldCheck className="h-3 w-3 mr-1" />
                    Vendor Mode Active
                  </Badge>
                )}
              </CardHeader>
              <CardContent className="flex-1 overflow-hidden p-0 border-t">
                <ChatPanel clientId={clientId} isVendorOnly={isVendorOnly} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="briefing" className="h-full m-0">
            <Card className="h-full flex flex-col">
              <CardHeader>
                <CardTitle>Dynamic Executive Briefings</CardTitle>
                <CardDescription>
                  Adaptive document summaries based on document type.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 overflow-hidden p-0 border-t">
                <BriefingPanel clientId={clientId} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="vendor" className="h-full m-0">
            <Card className="h-full flex flex-col">
              <CardHeader>
                <CardTitle>Vendor Intelligence Filter</CardTitle>
                <CardDescription>
                  Isolate retrieval for vendor-related procurement risks.
                </CardDescription>
              </CardHeader>
              <CardContent className="flex-1 p-6 space-y-8">
                <VendorToggle enabled={isVendorOnly} onToggle={setIsVendorOnly} />
                
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">What is Vendor Mode?</h3>
                  <p className="text-sm leading-relaxed">
                    Vendor Mode applies a strict metadata filter to all AI interactions. When active, the system will only 
                    retrieve context from documents marked as &quot;Vendor Documents&quot; during upload.
                  </p>
                  <ul className="text-sm space-y-2 list-disc pl-5">
                    <li>Isolate procurement risks in complex client vaults.</li>
                    <li>Ensure AI doesn't mix vendor terms with internal client policy.</li>
                    <li>Switch back to Global Mode anytime for full-vault intelligence.</li>
                  </ul>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </div>
      </Tabs>
    </div>
  );
}
