export type DocumentType = 'Contract' | 'Evidence' | 'Pleading' | 'Correspondence';

export interface BriefingSection {
  title: string;
  instruction: string;
}

export interface BriefingTemplate {
  documentType: DocumentType;
  sections: BriefingSection[];
}

export const BRIEFING_TEMPLATES: Record<DocumentType, BriefingTemplate> = {
  Contract: {
    documentType: 'Contract',
    sections: [
      { title: 'Parties', instruction: 'Identify all legal entities and individuals involved in the agreement.' },
      { title: 'Term', instruction: 'Specify the effective date, duration, and any renewal or termination provisions.' },
      { title: 'Key Obligations', instruction: 'Summarize the primary responsibilities and deliverables for each party.' }
    ]
  },
  Evidence: {
    documentType: 'Evidence',
    sections: [
      { title: 'Date of Incident', instruction: 'Identify the specific date and time the evidence pertains to.' },
      { title: 'Relevance', instruction: 'Explain why this document is significant to the case or matter.' },
      { title: 'Key Quotes', instruction: 'Extract the most impactful verbatim statements from the document.' }
    ]
  },
  Pleading: {
    documentType: 'Pleading',
    sections: [
      { title: 'Cause of Action', instruction: 'Summarize the legal grounds for the claim or defense.' },
      { title: 'Relief Sought', instruction: 'Identify the specific remedies or judgments requested from the court.' },
      { title: 'Key Allegations', instruction: 'List the primary factual assertions made in this document.' }
    ]
  },
  Correspondence: {
    documentType: 'Correspondence',
    sections: [
      { title: 'Sender/Recipient', instruction: 'Identify who sent the communication and to whom it was addressed.' },
      { title: 'Primary Subject', instruction: 'Summarize the main topic or purpose of the correspondence.' },
      { title: 'Action Items', instruction: 'List any requested tasks, deadlines, or follow-up requirements.' }
    ]
  }
};

export function getTemplateForType(type: string): BriefingTemplate {
  const docType = type as DocumentType;
  return BRIEFING_TEMPLATES[docType] || BRIEFING_TEMPLATES['Correspondence'];
}
