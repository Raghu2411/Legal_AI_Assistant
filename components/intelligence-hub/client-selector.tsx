'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Search, User, Lightbulb } from 'lucide-react';

interface Client {
  id: string;
  name: string;
  auto_case_id: string;
  case_type: string;
}

interface HubClientSelectorProps {
  clients: Client[];
  onSelect: (clientId: string) => void;
}

export const HubClientSelector: React.FC<HubClientSelectorProps> = ({ clients, onSelect }) => {
  const [selectedClient, setSelectedClient] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');

  const filteredClients = clients.filter(c => 
    c.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    c.auto_case_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="flex items-center justify-center min-h-full p-4 bg-muted/20">
      <Card className="w-full max-w-2xl shadow-lg border-primary/10">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold flex items-center gap-2">
            <Lightbulb className="h-6 w-6 text-primary" />
            Intelligence Hub Access
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Select a client to access cross-document intelligence, chat, and briefings.
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="space-y-3">
            <Label htmlFor="client-search">Search Client</Label>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                id="client-search"
                placeholder="Search by name or case ID..."
                className="pl-9"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-72 overflow-y-auto p-1 border rounded-md bg-background">
              {filteredClients.map(client => (
                <button
                  key={client.id}
                  onClick={() => setSelectedClient(client.id)}
                  className={`flex flex-col items-start p-3 text-left border rounded-lg transition-all ${
                    selectedClient === client.id 
                      ? 'border-primary bg-primary/5 ring-1 ring-primary' 
                      : 'hover:bg-accent'
                  }`}
                >
                  <span className="font-semibold text-sm flex items-center gap-1">
                    <User className="h-3 w-3" />
                    {client.name}
                  </span>
                  <span className="text-xs text-muted-foreground font-mono">
                    {client.auto_case_id} • {client.case_type}
                  </span>
                </button>
              ))}
              {filteredClients.length === 0 && (
                <div className="col-span-full py-12 text-center text-muted-foreground text-sm">
                  No clients found.
                </div>
              )}
            </div>
          </div>
        </CardContent>
        <CardFooter className="bg-accent/50 rounded-b-lg border-t pt-6">
          <Button 
            className="w-full" 
            size="lg" 
            disabled={!selectedClient}
            onClick={() => onSelect(selectedClient)}
          >
            Enter Intelligence Hub
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};
