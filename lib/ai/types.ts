export type ObligationStatus = 'pending' | 'confirmed' | 'rejected';
export type Classification = 'standard' | 'complex';

export interface TriageResult {
  classification: Classification;
  complexity_score: number; // 1-10
  reasoning: string;
  compliance_flags: ComplianceFlag[];
}

export interface ComplianceFlag {
  source: 'admin' | 'regulatory';
  status: 'passed' | 'failed';
  rule_violated?: string;
  description: string;
}

export interface ExtractedObligation {
  description: string;
  due_date: string | null; // ISO string or null
  is_tbd: boolean;
  metadata: Record<string, any>;
}
