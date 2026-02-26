import { z } from "zod"

// T006: Define Zod schema for client onboarding
export const clientSchema = z.object({
  name: z.string().min(2, "Name must be at least 2 characters"),
  case_type: z.string().min(2, "Case type must be at least 2 characters"),
})

export const editClientSchema = clientSchema.extend({
  status: z.enum(["Active", "Closed", "Archived"]),
})
