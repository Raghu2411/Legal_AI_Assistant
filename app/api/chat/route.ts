import { chatStream } from '@/lib/ai/chat-stream';

export async function POST(req: Request) {
  const { messages, clientId, isVendorOnly } = await req.json();
  
  if (!clientId) {
    return new Response('Missing clientId', { status: 400 });
  }

  return chatStream(messages, clientId, isVendorOnly);
}
