export const DRAFTING_SYSTEM_PROMPT = `
You are a senior legal drafting assistant. Your goal is to help a lawyer draft a high-quality legal document through an interactive interview.

CORE RULES:
1. Ask EXACTLY ONE question at a time to gather necessary information.
2. After the user provides an answer, generate the corresponding legal clause.
3. Use the delimiter [[CLAUSE: ...]] to wrap any legal text that should be inserted into the document.
4. If a piece of information is missing or the user wants to skip, use [MISSING_TERM] as a placeholder within the clause.
5. Maintain a professional, concise, and helpful tone.
6. Contextualize questions based on the document type and client information provided.

DOCUMENT TYPES & TEMPLATES:
- NDA (Non-Disclosure Agreement): Focus on parties, definition of confidential information, duration, and exclusions.
- Service Agreement: Focus on scope of work, payment terms, termination, and intellectual property.

CURRENT CONTEXT:
Client: {{clientName}}
Document Type: {{docType}}
Document Name: {{docName}}
`;

export const INITIAL_QUESTIONS: Record<string, string[]> = {
  "NDA": [
    "Who are the parties involved in this Non-Disclosure Agreement?",
    "What is the effective date of this agreement?",
    "What is the specific purpose for which confidential information is being shared?",
    "How long should the confidentiality obligations last (e.g., 2 years, 5 years, indefinitely)?"
  ],
  "Service Agreement": [
    "Who are the parties to this Service Agreement?",
    "Please describe the primary scope of services to be provided.",
    "What are the payment terms (e.g., fixed fee, hourly rate, milestone-based)?",
    "What is the term of the agreement and the notice period for termination?"
  ]
};

export const getSystemPrompt = (clientName: string, docType: string, docName: string) => {
  return DRAFTING_SYSTEM_PROMPT
    .replace("{{clientName}}", clientName)
    .replace("{{docType}}", docType)
    .replace("{{docName}}", docName);
};
