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
import { Download, Trash2, FileIcon, Clock, Eye, RefreshCw, Loader2, Sparkles } from "lucide-react"
import { deleteDocumentAction } from "@/lib/clients/document-actions"
import { useState, useEffect } from "react"
import { createClient } from "@/lib/supabase/client"
import { VectorStatusBadge, VectorStatus } from "@/components/ui/vector-status-badge"
import Link from "next/link"
import { revectorizeDocument } from "@/lib/review/actions"

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
  documents: initialDocuments,
  showReview = true,
  showTriage = true,
  showRevectorize = true
}: { 
  clientId: string
  documents: Document[]
  showReview?: boolean
  showTriage?: boolean
  showRevectorize?: boolean
}) {
  const [documents, setDocuments] = useState(initialDocuments)
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const [revectorizingId, setRevectorizingId] = useState<string | null>(null)
  const [triagingId, setTriagingId] = useState<string | null>(null)
  const supabase = createClient()

  useEffect(() => {
    setDocuments(initialDocuments)
  }, [initialDocuments])

  const handleTriage = async (docId: string) => {
    setTriagingId(docId)
    try {
      const response = await fetch("/api/triage/process", {
        method: "POST",
        body: JSON.stringify({ documentId: docId }),
        headers: { "Content-Type": "application/json" }
      })
      const result = await response.json()
      if (result.success) {
        alert("Triage and extraction complete!")
        window.location.reload()
      } else {
        alert(`Triage failed: ${result.error}`)
      }
    } catch (err) {
      alert("An error occurred during triage.")
    } finally {
      setTriagingId(null)
    }
  }

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

  const handleRevectorize = async (docId: string) => {
    if (!confirm("This will replace current search chunks with your refined version from Review Studio. Proceed?")) return
    
    setRevectorizingId(docId)
    try {
      const result = await revectorizeDocument(docId)
      if (result.success) {
        alert("Re-vectorization started. Status will update shortly.")
      } else {
        alert(result.error)
      }
    } catch (err) {
      alert("An error occurred during re-vectorization.")
    } finally {
      setRevectorizingId(null)
    }
  }

  return (
    <div className="w-full overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="min-w-[200px]">File Name</TableHead>
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
                  <FileIcon className="h-4 w-4 text-muted-foreground shrink-0" />
                  <span className="truncate max-w-[150px] md:max-w-none" title={doc.file_name}>
                    {doc.file_name}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <Badge variant="outline">{doc.doc_type}</Badge>
              </TableCell>
              <TableCell className="text-muted-foreground text-sm whitespace-nowrap">
                <div className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {new Date(doc.uploaded_at).toLocaleDateString()}
                </div>
              </TableCell>
              <TableCell>
                <VectorStatusBadge status={doc.vector_status as VectorStatus} />
              </TableCell>
              <TableCell className="text-right">
                <div className="flex justify-end gap-1 md:gap-2">
                  {showTriage && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="gap-1 md:gap-2 h-8 border-primary/50 text-primary hover:bg-primary/5"
                      onClick={() => handleTriage(doc.id)}
                      disabled={triagingId === doc.id}
                    >
                      {triagingId === doc.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <Sparkles className="h-3.5 w-3.5" />
                      )}
                      <span className="hidden sm:inline">Triage Scan</span>
                    </Button>
                  )}
                  {showRevectorize && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="gap-1 md:gap-2 h-8"
                      onClick={() => handleRevectorize(doc.id)}
                      disabled={revectorizingId === doc.id}
                    >
                      {revectorizingId === doc.id ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                      ) : (
                        <RefreshCw className="h-3.5 w-3.5" />
                      )}
                      <span className="hidden sm:inline">Re-vectorize</span>
                    </Button>
                  )}
                  {showReview && (
                    <Button 
                      variant="outline" 
                      size="sm"
                      className="gap-1 md:gap-2 h-8"
                      asChild
                    >
                      <Link href={`/review/${doc.id}`}>
                        <Eye className="h-3.5 w-3.5" />
                        <span className="hidden sm:inline">Review</span>
                      </Link>
                    </Button>
                  )}
                  <Button 
                    variant="ghost" 
                    size="icon"
                    className="h-8 w-8"
                    onClick={() => handleDownload(doc.file_url, doc.file_name)}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                  <Button 
                    variant="ghost" 
                    size="icon"
                    className="h-8 w-8 text-destructive hover:text-destructive hover:bg-destructive/10"
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
  );
}
