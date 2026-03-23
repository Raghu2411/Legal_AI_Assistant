import { Groq } from 'groq-sdk';
import { OpenAIStream, StreamingTextResponse, StreamData } from 'ai';
import { retrieveContextV2 } from '../supabase/vector-queries';
import { EMBEDDING_MODEL, mxbai } from './mixedbread';

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

export async function chatStream(
  messages: any[],
  clientId: string,
  isVendorOnly: boolean = false
) {
  const lastMessage = messages[messages.length - 1].content;

  // 1. Generate embedding for query
  const embeddingResponse = await mxbai.embeddings.create({
    model: EMBEDDING_MODEL,
    input: [lastMessage],
    encoding_format: 'float'
  });

  const queryEmbedding = embeddingResponse.data[0].embedding as number[];

  // 2. Retrieve context
  const context = await retrieveContextV2({
    queryEmbedding,
    targetClientId: clientId,
    isVendorOnly,
    matchCount: 5
  });

  const contextText = context
    .map((c: any, i: number) => `[Source ${i + 1}]: ${c.content}`)
    .join('\n\n');

  const systemPrompt = `You are a legal AI assistant. Use the following pieces of context to answer the user's question. 
If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.
CRITICAL: You MUST include numbered citations like [1], [2], etc., when referring to specific information from the sources.
The sources are provided below as [Source 1], [Source 2], etc. Map your citations [1] to [Source 1], [2] to [Source 2], and so on.

CONTEXT:
${contextText}

User Question: ${lastMessage}`;

  // 3. Create stream
  const response = await groq.chat.completions.create({
    model: 'llama-3.3-70b-versatile',
    messages: [
      { role: 'system', content: systemPrompt },
      ...messages.slice(0, -1),
      { role: 'user', content: lastMessage }
    ],
    stream: true,
  });

  const data = new StreamData();
  
  // Add citation metadata to the stream
  data.append({
    citations: context.map((c: any, i: number) => ({
      index: i + 1,
      content: c.content,
      document_id: c.document_id,
      metadata: c.metadata
    }))
  });

  const stream = OpenAIStream(response as any, {
    onFinal() {
      data.close();
    },
  });

  return new StreamingTextResponse(stream, {}, data);
}
