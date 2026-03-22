import { draftingChatStream } from "@/lib/ai/drafting-orchestrator";
import { createClient } from "@/lib/supabase/server";

export async function POST(req: Request) {
  try {
    const { messages, session, clientName } = await req.json();
    
    const supabase = createClient();
    const { data: { user } } = await supabase.auth.getUser();

    if (!user) {
      return new Response("Unauthorized", { status: 401 });
    }

    // Verify session belongs to the user or is valid
    // For now, we trust the session object since it's just for the drafting session state

    return await draftingChatStream(messages, session, clientName);
  } catch (error: any) {
    console.error("Drafting chat error:", error);
    return new Response(JSON.stringify({ error: error.message }), { 
      status: 500,
      headers: { 'Content-Type': 'application/json' }
    });
  }
}
