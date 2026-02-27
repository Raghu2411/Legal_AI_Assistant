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
import { Download, Trash2, FileIcon, Clock } from "lucide-react"
import { deleteDocumentAction } from "@/lib/clients/actions"
import { useState } from "react"
import { createClient } from "@/lib/supabase/client"
import { VectorStatusBadge, VectorStatus } from "@/components/ui/vector-status-badge"

interface Document {
  id: string
  file_url: string
  file_name: string
  doc_type: string
  uploaded_at: string
  vector_status?: string
}

export function VaultView({ 
  clientId, 
  documents: initialDocuments 
}: { 
  clientId: string
  documents: Document[] 
}) {
  const [documents, setDocuments] = useState(initialDocuments)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const supabase = createClient()

  const handleDelete = async (docId: string, fileUrl: string) => {
    if (!confirm("Are you sure you want to delete this document?")) return
    
    setDeletingId(docId)
    const result = await deleteDocumentAction(docId, clientId, fileUrl)
    
    if (result.success) {
      setDocuments(documents.filter(d => d.id !== docId))
    } else {
      alert(`Error deleting document: ${result.error}`)
    }
    setDeletingId(null)
  }

  const handleDownload = async (fileUrl: string, fileName: string) => {
    const { data, error } = await supabase.storage
      .from("client-vaults")
      .download(fileUrl)

    if (error) {
      alert(`Error downloading file: ${error.message}`)
      return
    }

    const url = URL.createObjectURL(data)
    const a = document.createElement("a")
    a.href = url
    a.download = fileName
    document.body.appendChild(a)
    a.click()
    URL.revokeObjectURL(url)
    document.body.removeChild(a)
  }

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>File Name</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Uploaded</TableHead>
            <TableHead>Status</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {documents.map((doc) => (
            <TableRow key={doc.id}>
              <TableCell className="font-medium">
                <div className="flex items-center gap-2">
                  <FileIcon className="h-4 w-4 text-muted-foreground" />
                  {doc.file_name}
                </div>
              </TableCell>
              <TableCell>
                <Badge variant="outline">{doc.doc_type}</Badge>
              </TableCell>
              <TableCell className="text-muted-foreground text-sm">
                <div className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {new Date(doc.uploaded_at).toLocaleDateString()}
                </div>
              </TableCell>
              <TableCell>
                <VectorStatusBadge status={doc.vector_status as VectorStatus} />
              </TableCell>
              <TableCell className="text-right">
                <div className="flex justify-end gap-2">
                  <Button 
                    variant="ghost" 
                    size="icon"
                    onClick={() => handleDownload(doc.file_url, doc.file_name)}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="icon"
                    className="text-destructive hover:text-destructive hover:bg-destructive/10"
                    disabled={deletingId === doc.id}
                    onClick={() => handleDelete(doc.id, doc.file_url)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </TableCell>
            </TableRow>
          ))}
          {documents.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="h-24 text-center text-muted-foreground">
                No documents in vault.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
