import { Groq } from "groq-sdk";

export const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

export const DEFAULT_MODEL = "llama-3.3-70b-versatile";

export async function getGroqCompletion(systemPrompt: string, userPrompt: string, jsonMode = true) {
  try {
    const response = await groq.chat.completions.create({
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      model: DEFAULT_MODEL,
      temperature: 0.1,
      response_format: jsonMode ? { type: "json_object" } : undefined,
    });

    const content = response.choices[0]?.message?.content || "";
    return jsonMode ? JSON.parse(content) : content;
  } catch (error) {
    console.error("Groq Completion Error:", error);
    throw error;
  }
}
