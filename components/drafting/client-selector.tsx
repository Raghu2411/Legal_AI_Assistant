'use client';

import React, { useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Search, FilePlus, User } from 'lucide-react';

interface Client {
  id: string;
  name: string;
  auto_case_id: string;
  case_type: string;
}

interface ClientSelectorProps {
  clients: Client[];
  onStart: (clientId: string, docType: string, docName: string) => void;
}

const DOCUMENT_TYPES = [
  { id: 'NDA', name: 'Non-Disclosure Agreement', icon: '🔒' },
  { id: 'Service Agreement', name: 'Service Agreement', icon: '📝' },
];

export const ClientSelector: React.FC<ClientSelectorProps> = ({ clients, onStart }) => {
  const [selectedClient, setSelectedClient] = useState<string>('');
  const [selectedDocType, setSelectedDocType] = useState<string>('');
  const [docName, setDocName] = useState<string>('');
  const [searchTerm, setSearchTerm] = useState<string>('');

  const filteredClients = clients.filter(c => 
    c.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
    c.auto_case_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const canStart = selectedClient && selectedDocType && docName.trim();

  const handleStart = () => {
    if (canStart) {
      onStart(selectedClient, selectedDocType, docName);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-full p-4">
      <Card className="w-full max-w-2xl shadow-lg border-primary/10">
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl font-bold flex items-center gap-2">
            <FilePlus className="h-6 w-6 text-primary" />
            New Drafting Session
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Select a client and document type to begin the AI-assisted drafting process.
          </p>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Client Selection */}
          <div className="space-y-3">
            <Label htmlFor="client-search">1. Select Client</Label>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                id="client-search"
                placeholder="Search clients..."
                className="pl-9"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-48 overflow-y-auto p-1 border rounded-md">
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
                <div className="col-span-full py-8 text-center text-muted-foreground text-sm">
                  No clients found matching your search.
                </div>
              )}
            </div>
          </div>

          {/* Document Type Selection */}
          <div className="space-y-3">
            <Label>2. Document Type</Label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {DOCUMENT_TYPES.map(type => (
                <button
                  key={type.id}
                  onClick={() => {
                    setSelectedDocType(type.id);
                    if (!docName) setDocName(`${type.id} - ${new Date().toLocaleDateString()}`);
                  }}
                  className={`flex items-center gap-3 p-4 border rounded-lg transition-all ${
                    selectedDocType === type.id 
                      ? 'border-primary bg-primary/5 ring-1 ring-primary' 
                      : 'hover:bg-accent'
                  }`}
                >
                  <span className="text-2xl">{type.icon}</span>
                  <div className="flex flex-col text-left">
                    <span className="font-semibold text-sm">{type.name}</span>
                    <span className="text-xs text-muted-foreground">Standard template</span>
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Document Name */}
          <div className="space-y-2">
            <Label htmlFor="doc-name">3. Document Name</Label>
            <Input
              id="doc-name"
              placeholder="e.g. Mutual NDA - Acme Corp"
              value={docName}
              onChange={(e) => setDocName(e.target.value)}
            />
          </div>
        </CardContent>
        <CardFooter className="bg-accent/50 rounded-b-lg border-t pt-6">
          <Button 
            className="w-full" 
            size="lg" 
            disabled={!canStart}
            onClick={handleStart}
          >
            Start Drafting Session
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
};
