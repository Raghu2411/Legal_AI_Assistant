"use client"

import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import { clientSchema } from "@/lib/clients/schemas"
import { z } from "zod"
import { Button } from "@/components/ui/button"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { createClientAction } from "@/lib/clients/actions"
import { useState } from "react"
import { useRouter } from "next/navigation"

type ClientFormValues = z.infer<typeof clientSchema>

export function ClientForm() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  const form = useForm<ClientFormValues>({
    resolver: zodResolver(clientSchema),
    defaultValues: {
      name: "",
      case_type: "",
    },
  })

  async function onSubmit(values: ClientFormValues) {
    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append("name", values.name)
    formData.append("case_type", values.case_type)

    const result = await createClientAction(formData)

    if (result.error) {
      setError(result.error)
      setLoading(false)
    } else {
      router.push("/clients")
      router.refresh()
    }
  }

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
        {error && (
          <div className="p-3 bg-destructive/10 text-destructive text-sm rounded-md">
            {error}
          </div>
        )}
        
        <FormField
          control={form.control}
          name="name"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Client Name</FormLabel>
              <FormControl>
                <Input placeholder="e.g. Acme Corp" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <FormField
          control={form.control}
          name="case_type"
          render={({ field }) => (
            <FormItem>
              <FormLabel>Case Type</FormLabel>
              <FormControl>
                <Input placeholder="e.g. Corporate Law" {...field} />
              </FormControl>
              <FormMessage />
            </FormItem>
          )}
        />

        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? "Adding Client..." : "Add Client"}
        </Button>
      </form>
    </Form>
  )
}
