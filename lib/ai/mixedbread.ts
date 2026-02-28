import Mixedbread from '@mixedbread/sdk';

if (!process.env.MIXEDBREAD_API_KEY) {
  throw new Error("MIXEDBREAD_API_KEY is not set in environment variables.");
}

export const mxbai = new Mixedbread({
  apiKey: process.env.MIXEDBREAD_API_KEY,
});

export const EMBEDDING_MODEL = 'mixedbread-ai/mxbai-embed-large-v1';
