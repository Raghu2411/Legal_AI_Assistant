# Contract: Intelligence Hub UI

## Interface: Chat UI Context

```typescript
interface Citation {
  id: number;
  filename: string;
  pageNumber: number;
  snippet: string;
}

interface ChatResponseMetadata {
  citations: Citation[];
}

// Handled via Vercel AI SDK 'onFinish' callback to update citation sidebar
```

## Interface: Dynamic Briefing Structure

```typescript
const BriefingTemplates = {
  Contract: [
    { title: "Parties", prompt: "Identify the primary legal entities involved." },
    { title: "Term", prompt: "Specify the duration and renewal clauses." },
    { title: "Key Obligations", prompt: "Summarize the top 3 duties per party." }
  ],
  Evidence: [
    { title: "Date of Incident", prompt: "Identify the exact date or period described." },
    { title: "Relevance", prompt: "How does this document impact the case?" },
    { title: "Key Quotes", prompt: "Extract 2-3 critical verbatim quotes." }
  ],
  Pleading: [
    { title: "Claims", prompt: "List the formal legal causes of action." },
    { title: "Parties", prompt: "Who are the named Plaintiff and Defendant?" },
    { title: "Relief", prompt: "What specific remedy is being sought?" }
  ],
  Correspondence: [
    { title: "Sender", prompt: "Who authored this communication?" },
    { title: "Recipient", prompt: "To whom was this addressed?" },
    { title: "Key Demand", prompt: "What is the primary request or assertion?" }
  ]
};
```
