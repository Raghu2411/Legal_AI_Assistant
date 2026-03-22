"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { AlertCircle, CheckCircle2 } from "lucide-react";
import { TriageOverrideModal } from "./triage-override-modal";

interface Document {
  id: string;
  file_name: string;
  classification: "standard" | "complex";
  complexity_score: number;
  uploaded_at: string;
}

export function TriageTable({ initialDocuments }: { initialDocuments: Document[] }) {
  const [documents, setDocuments] = useState(initialDocuments);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);

  useEffect(() => {
    setDocuments(initialDocuments);
  }, [initialDocuments]);

  const handleOverride = (doc: Document) => {
    setSelectedDoc(doc);
  };

  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Document Name</TableHead>
            <TableHead>Classification</TableHead>
            <TableHead>Complexity</TableHead>
            <TableHead>Uploaded At</TableHead>
            <TableHead className="text-right">Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {documents.map((doc) => (
            <TableRow key={doc.id}>
              <TableCell className="font-medium">{doc.file_name}</TableCell>
              <TableCell>
                <Badge
                  variant={doc.classification === "complex" ? "destructive" : "secondary"}
                >
                  {doc.classification.toUpperCase()}
                </Badge>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <span className="text-sm">{doc.complexity_score}/10</span>
                  {doc.complexity_score >= 7 ? (
                    <AlertCircle className="h-4 w-4 text-destructive" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-green-500" />
                  )}
                </div>
              </TableCell>
              <TableCell>{new Date(doc.uploaded_at).toLocaleDateString()}</TableCell>
              <TableCell className="text-right">
                <Button variant="ghost" size="sm" onClick={() => handleOverride(doc)}>
                  Override
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {selectedDoc && (
        <TriageOverrideModal
          document={selectedDoc}
          onClose={() => setSelectedDoc(null)}
          onSuccess={(updatedDoc: Document) => {
            setDocuments(prev => prev.map(d => d.id === updatedDoc.id ? updatedDoc : d));
            setSelectedDoc(null);
          }}
        />
      )}
    </div>
  );
}
