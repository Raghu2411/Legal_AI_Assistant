# Legal_AI_Assistant Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-03-20

## Active Technologies
- Python 3.11, FastAPI, TypeScript / Next.js 14+ (App Router), Supabase (Auth, DB, Storage, pgvector), Vercel AI SDK, Groq SDK (Llama 3.3), shadcn/ui, TipTap

## Project Structure
- `app/`: Next.js App Router routes
- `components/`: Reusable UI components
- `lib/`: Shared logic, AI services, Supabase clients
- `supabase/`: Database migrations and functions
- `specs/`: Feature specifications and plans

## Active Features
- **001-auth-rbac-setup**: RBAC (Admin, Lawyer, Client) via Supabase Auth
- **002-admin-crud-console**: Admin dashboard for user/playbook management
- **003-client-management**: Lawyer-facing client and document vault management
- **004-rag-infrastructure**: Mixedbread-ai embeddings + pgvector retrieval
- **005-ai-contract-review**: Risk analysis studio with TipTap integration
- **006-intelligence-hub**: Tabbed AI Hub (Chat with citations, Dynamic Briefings, Vendor Mode)

## Commands
- `npm run dev`: Start development server
- `npm run build`: Build for production
- `npm run lint`: Run ESLint
- `npx supabase migration new <name>`: Create new migration
- `npx supabase db push`: Push migrations to local DB

## Code Style
- TypeScript / Next.js 14+ (App Router): Follow standard conventions
- Constitution Principles apply to all AI-related features

## Recent Changes
- 006-intelligence-hub: Added Tabbed AI Hub with Vercel AI SDK, citation-aware Chat, Dynamic Briefings, and Metadata-filtered Vendor Mode.
- 005-ai-contract-review: Added Risk analysis studio with TipTap integration.
- 004-rag-infrastructure: Initialized RAG infrastructure with Mixedbread-ai.



<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
