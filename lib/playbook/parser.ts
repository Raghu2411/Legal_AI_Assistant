import mammoth from 'mammoth';

/**
 * Extracts text from a file buffer (PDF or DOCX).
 */
export async function extractTextFromFile(buffer: Buffer, fileType: string): Promise<string> {
  console.log('extractTextFromFile: Starting extraction, buffer size:', buffer.length, 'Type:', fileType);
  
  try {
    if (fileType === 'application/pdf' || fileType.endsWith('.pdf')) {
      // PDF Parsing Logic using classic pdf-parse (v1.1.1)
      // We use a dynamic require to avoid bundling issues
      const pdf = require('pdf-parse');
      
      console.log('extractTextFromFile: Calling pdf-parse (v1.1.1)...');
      const data = await pdf(buffer);
      
      const extractedText = data?.text || "";
      console.log('extractTextFromFile: PDF extraction complete. Text length:', extractedText.length);
      return extractedText;
    } 
    
    if (fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || fileType.endsWith('.docx')) {
      // DOCX Parsing Logic using Mammoth
      const result = await mammoth.extractRawText({ buffer });
      const extractedText = result.value || "";
      console.log('extractTextFromFile: Successfully extracted DOCX text, length:', extractedText.length);
      return extractedText;
    }

    if (fileType === 'text/plain' || fileType.endsWith('.txt')) {
      const extractedText = buffer.toString('utf8');
      console.log('extractTextFromFile: Successfully read TXT text, length:', extractedText.length);
      return extractedText;
    }

    throw new Error(`Unsupported file type: ${fileType}`);
  } catch (error: any) {
    console.error('Error parsing file details:', {
      message: error.message,
      stack: error.stack
    });
    throw new Error(`Failed to extract text from ${fileType}: ${error.message}`);
  }
}

/**
 * Normalizes text for AI context.
 */
export function normalizeContext(text: string): string {
  if (!text) return "";
  return text
    .replace(/\s+/g, ' ')
    .replace(/\n+/g, ' ')
    .trim();
}
