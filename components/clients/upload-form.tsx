"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { uploadDocumentAction } from "@/lib/clients/document-actions"
import { Upload, AlertCircle, FileCheck, ShieldCheck } from "lucide-react"

export function UploadForm({ clientId }: { clientId: string }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setSuccess(false)

    const formData = new FormData(e.currentTarget)
    const result = await uploadDocumentAction(clientId, formData)

    if (result.error) {
      setError(result.error)
      setLoading(false)
    } else {
      setSuccess(true)
      setLoading(false)
      // Reset form
      ;(e.target as HTMLFormElement).reset()
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="space-y-2">
        <Label htmlFor="file">File (PDF, DOCX, TXT)</Label>
        <Input 
          id="file" 
          name="file" 
          type="file" 
          required 
          accept=".pdf,.docx,.txt"
          className="cursor-pointer"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="docType">Document Type</Label>
        <select
          id="docType"
          name="docType"
          required
          className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <option value="">Select type...</option>
          <option value="Contract">Contract</option>
          <option value="Evidence">Evidence</option>
          <option value="Correspondence">Correspondence</option>
          <option value="Pleading">Pleading</option>
        </select>
      </div>

      <div className="flex items-center space-x-2 p-2 rounded-md border bg-muted/20">
        <input
          type="checkbox"
          id="isVendor"
          name="isVendor"
          className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
        />
        <Label htmlFor="isVendor" className="flex items-center gap-1.5 cursor-pointer">
          <ShieldCheck className="h-4 w-4 text-primary" />
          Mark as Vendor Document
        </Label>
      </div>

      {error && (
        <div className="flex items-center gap-2 p-3 bg-destructive/10 text-destructive text-sm rounded-md">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      {success && (
        <div className="flex items-center gap-2 p-3 bg-green-500/10 text-green-500 text-sm rounded-md">
          <FileCheck className="h-4 w-4" />
          Document uploaded successfully!
        </div>
      )}

      <Button type="submit" className="w-full" disabled={loading}>
        <Upload className="h-4 w-4 mr-2" />
        {loading ? "Uploading..." : "Upload Document"}
      </Button>
    </form>
  )
}
