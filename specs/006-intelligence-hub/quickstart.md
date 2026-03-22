# Quickstart: Intelligence Hub Manual Verification

**Status**: Manual Only (Principle III compliance)

## Pre-requisites
1. A client with multiple documents uploaded.
2. At least one document tagged as 'Vendor Document' at upload.
3. Documents must have 'Completed' vectorization status (Step 4 infrastructure).

## Verification Steps

### 1. Tab Navigation & UI
- Navigate to the Client Detail page.
- Select the 'Intelligence Hub' tab.
- **PASS**: Tabs for 'Chat', 'Briefing', and 'Vendor Mode' render correctly.

### 2. Conversational Chat & Session Memory
- In 'Chat' tab, ask "What is the primary concern in the NDAs?"
- Follow up with "Is there a specific clause mentioned for this?"
- **PASS**: AI understands "this" refers to the previous context.
- Refresh the page.
- **PASS**: Chat history is cleared (FR-002 Volatile In-Memory only).

### 3. Citations & Footnotes
- Send any query to the Chat.
- **PASS**: Response includes `[1]` style footnotes.
- Click a footnote badge.
- **PASS**: The UI scrolls/highlights the document segment in the viewer.

### 4. Dynamic Briefing
- Open the 'Briefing' tab for a 'Contract' type document.
- Verify sections: Parties, Term, Key Obligations.
- Switch to an 'Evidence' type document.
- Verify sections: Date of Incident, Relevance, Key Quotes.
- **PASS**: Sections adapt dynamically (FR-006).

### 5. Vendor Mode Filtering
- Toggle 'Vendor Mode' to ON.
- Ask "What are the procurement costs?"
- **PASS**: AI only uses context from documents where `is_vendor = true`.
- **PASS**: If no vendor documents exist or match, LLM handles graceful refusal.
