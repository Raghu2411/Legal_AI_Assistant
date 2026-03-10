export async function getTextSplitter(chunkSize = 500, chunkOverlap = 50) {
  const { RecursiveCharacterTextSplitter } = await import('@langchain/textsplitters');
  return new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });
}
