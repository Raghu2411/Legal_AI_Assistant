import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';

export function getTextSplitter(chunkSize = 500, chunkOverlap = 50) {
  return new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });
}
