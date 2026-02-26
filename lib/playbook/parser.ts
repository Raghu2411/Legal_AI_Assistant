import mammoth from 'mammoth';

/**
 * Extracts text from a file buffer (PDF or DOCX).
 */
export async function extractTextFromFile(buffer: Buffer, fileType: string): Promise<string> {
  console.log('extractTextFromFile: Starting extraction, buffer size:', buffer.length, 'Type:', fileType);
  
  try {
    if (fileType === 'application/pdf' || fileType.endsWith('.pdf')) {
      // PDF Parsing Logic
      let pdf;
      try {
        pdf = require('pdf-parse');
      } catch (e) {
        console.warn('pdf-parse primary require failed, trying lib/pdf-parse.js');
        pdf = require('pdf-parse/lib/pdf-parse.js');
      }
      
      let parseFunc = pdf;
      if (typeof pdf !== 'function') {
        if (pdf.default && typeof pdf.default === 'function') {
          parseFunc = pdf.default;
        } else if (pdf.PDFParse && typeof pdf.PDFParse === 'function') {
          parseFunc = pdf.PDFParse;
        }
      }

      if (typeof parseFunc !== 'function') {
        throw new Error(`pdf-parse is not a function (type: ${typeof parseFunc})`);
      }
      
      const data = await parseFunc(buffer);
      return data.text || "";
    } 
    
    if (fileType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' || fileType.endsWith('.docx')) {
      // DOCX Parsing Logic using Mammoth
      const result = await mammoth.extractRawText({ buffer });
      console.log('extractTextFromFile: Successfully extracted DOCX text, length:', result.value?.length);
      return result.value;
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
  return text
    .replace(/\s+/g, ' ')
    .replace(/\n+/g, ' ')
    .trim();
}
