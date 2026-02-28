"use client"

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Eye, FileText, Search } from "lucide-react"
import Link from "next/link"
import { useState } from "react"
import { Input } from "@/components/ui/input"
import { ClientEditModal } from "./client-edit-modal"

interface Client {
  id: string
  auto_case_id: string
  name: string
  case_type: string
  status: string
  created_at: string
  profiles?: {
    full_name: string
  }
}

interface ClientTableProps {
  initialClients: any[]
  isAdmin?: boolean
}

export function ClientTable({ initialClients, isAdmin = false }: ClientTableProps) {
  const [search, setSearch] = useState("")
  
  const filteredClients = initialClients.filter(client => 
    client.name.toLowerCase().includes(search.toLowerCase()) ||
    client.auto_case_id.toLowerCase().includes(search.toLowerCase()) ||
    (isAdmin && client.profiles?.full_name?.toLowerCase().includes(search.toLowerCase()))
  )

  const basePath = isAdmin ? "/admin/clients" : "/clients"

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 max-w-sm">
        <Search className="h-4 w-4 text-muted-foreground" />
        <Input 
          placeholder="Search by name or case ID..." 
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Case ID</TableHead>
              <TableHead>Client Name</TableHead>
              <TableHead>Case Type</TableHead>
              {isAdmin && <TableHead>Lawyer</TableHead>}
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredClients.map((client) => (
              <TableRow key={client.id}>
                <TableCell className="font-mono font-medium">{client.auto_case_id}</TableCell>
                <TableCell>{client.name}</TableCell>
                <TableCell>{client.case_type}</TableCell>
                {isAdmin && <TableCell>{client.profiles?.full_name}</TableCell>}
                <TableCell>
                  <Badge variant={client.status === 'Active' ? 'default' : 'secondary'}>
                    {client.status}
                  </Badge>
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex justify-end gap-2">
                    {isAdmin && <ClientEditModal client={client} />}
                    <Button variant="outline" size="sm" asChild>
                      <Link href={`${basePath}/${client.id}`}>
                        <Eye className="h-4 w-4 mr-2" />
                        View
                      </Link>
                    </Button>
                    <Button variant="outline" size="sm" asChild>
                      <Link href={`${basePath}/${client.id}/vault`}>
                        <FileText className="h-4 w-4 mr-2" />
                        Vault
                      </Link>
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
            {filteredClients.length === 0 && (
              <TableRow>
                <TableCell colSpan={isAdmin ? 6 : 5} className="h-24 text-center">
                  No clients found.
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
