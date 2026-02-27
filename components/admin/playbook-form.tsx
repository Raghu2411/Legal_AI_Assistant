"use client"

import { useState } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import * as z from "zod"
import { Button } from "@/components/ui/button"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { 
  Card, 
  CardContent, 
  CardDescription, 
  CardFooter, 
  CardHeader, 
  CardTitle 
} from "@/components/ui/card"
import { uploadPlaybook, updateGoldenRules } from "@/app/(admin)/admin/playbook/actions"
import { FileUp, Save, CheckCircle2, AlertCircle } from "lucide-react"

const playbookSchema = z.object({
  file: z.any().optional(),
  golden_rules: z.string().min(10, {
    message: "Golden Rules must be at least 10 characters.",
  }),
})

export function PlaybookForm({ initialRules, currentVersion }: { initialRules: string, currentVersion: number }) {
  const [isUploading, setIsUploading] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  const form = useForm<z.infer<typeof playbookSchema>>({
    resolver: zodResolver(playbookSchema),
    defaultValues: {
      golden_rules: initialRules,
    },
  })

  async function onUpload(values: z.infer<typeof playbookSchema>) {
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    const file = fileInput?.files?.[0]

    if (!file) {
      setMessage({ type: 'error', text: "Please select a file to upload." })
      return
    }

    setIsUploading(true)
    setMessage(null)

    const formData = new FormData()
    formData.append("file", file)
    formData.append("golden_rules", values.golden_rules)

    const result = await uploadPlaybook(formData)
    
    if (result.success) {
      setMessage({ type: 'success', text: "Playbook uploaded and parsed successfully." })
      form.reset({ golden_rules: values.golden_rules })
    } else {
      setMessage({ type: 'error', text: result.error || "Upload failed." })
    }
    
    setIsUploading(false)
  }

  async function onSaveRules(values: z.infer<typeof playbookSchema>) {
    setIsUploading(true)
    setMessage(null)

    const result = await updateGoldenRules(values.golden_rules)
    
    if (result.success) {
      setMessage({ type: 'success', text: "Golden Rules updated successfully." })
    } else {
      setMessage({ type: 'error', text: result.error || "Save failed." })
    }
    
    setIsUploading(false)
  }

  return (
    <Form {...form}>
      <form className="space-y-8">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileUp className="h-5 w-5" />
              Legal Playbook (PDF/DOCX)
            </CardTitle>
            <CardDescription>
              Upload the firm&apos;s detailed legal playbook. This will be parsed and used as AI context.
              Supports PDF and DOCX formats.
              Current Version: <span className="font-bold text-primary">v{currentVersion}</span>
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <Input id="playbook" type="file" accept=".pdf,.docx" className="cursor-pointer" />
            </div>
          </CardContent>
          <CardFooter className="border-t bg-muted/50 px-6 py-4">
            <Button 
              type="button" 
              onClick={form.handleSubmit(onUpload)} 
              disabled={isUploading}
            >
              {isUploading ? "Processing..." : "Upload & Update Version"}
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Save className="h-5 w-5" />
              Golden Rules
            </CardTitle>
            <CardDescription>
              High-level firm principles that take priority in AI responses.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <FormField
              control={form.control}
              name="golden_rules"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <Textarea
                      placeholder="Enter the firm&apos;s Golden Rules..."
                      className="min-h-[200px] font-mono text-sm"
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
          </CardContent>
          <CardFooter className="border-t bg-muted/50 px-6 py-4 flex justify-between items-center">
            <Button 
              type="button" 
              variant="outline"
              onClick={form.handleSubmit(onSaveRules)} 
              disabled={isUploading}
            >
              Save Text Only
            </Button>

            {message && (
              <div className={`flex items-center gap-2 text-sm font-medium ${
                message.type === 'success' ? 'text-green-600' : 'text-destructive'
              }`}>
                {message.type === 'success' ? <CheckCircle2 className="h-4 w-4" /> : <AlertCircle className="h-4 w-4" />}
                {message.text}
              </div>
            )}
          </CardFooter>
        </Card>
      </form>
    </Form>
  )
}
