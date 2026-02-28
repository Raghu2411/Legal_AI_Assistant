# Legal_AI_Assistant Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-02-25

## Active Technologies
- [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION] + [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION] (002-admin-crud-console)
- [if applicable, e.g., PostgreSQL, CoreData, files or N/A] (002-admin-crud-console)
- TypeScript / Next.js 14+ (App Router) + Supabase (Auth, DB, Storage), shadcn/ui, react-hook-form, zod (003-client-management)
- PostgreSQL (`clients`, `documents`), Supabase Storage (`client-vaults`) (003-client-management)
- TypeScript / Next.js 14+ (App Router) + `mixedbread-ai` SDK, `langchain` (RecursiveCharacterTextSplitter), `pdf-parse`, `mammoth` (for DOCX) (004-rag-infrastructure)
- Supabase PostgreSQL (pgvector), Supabase Storage (004-rag-infrastructure)

- TypeScript / Next.js 14+ (App Router) + @supabase/auth-helpers-nextjs, @supabase/supabase-js, tailwindcss, lucide-react, shadcn/ui, react-hook-form, zod (001-auth-rbac-setup)

## Project Structure

```text
backend/
frontend/
tests/
```

## Commands

npm test; npm run lint

## Code Style

TypeScript / Next.js 14+ (App Router): Follow standard conventions

## Recent Changes
- 004-rag-infrastructure: Added TypeScript / Next.js 14+ (App Router) + `mixedbread-ai` SDK, `langchain` (RecursiveCharacterTextSplitter), `pdf-parse`, `mammoth` (for DOCX)
- 003-client-management: Added TypeScript / Next.js 14+ (App Router) + Supabase (Auth, DB, Storage), shadcn/ui, react-hook-form, zod
- 003-client-management: Added [e.g., Python 3.11, Swift 5.9, Rust 1.75 or NEEDS CLARIFICATION] + [e.g., FastAPI, UIKit, LLVM or NEEDS CLARIFICATION]


<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
