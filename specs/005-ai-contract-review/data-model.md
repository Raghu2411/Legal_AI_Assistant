# Data Model: AI Contract Review (Step 5)

## Entities

### RiskAnalysis
Represents a single AI scan session for a document.

- **id** (uuid, primary key)
- **document_id** (uuid, foreign key to `documents.id`)
- **timestamp** (timestamptz, default now())
- **version** (int, auto-incrementing for document)
- **status** (enum: `pending`, `completed`, `failed`)
- **raw_json** (jsonb, stores the complete AI response for audit)

### ClauseAnalysis
Represents an individual clause's assessment within a scan.

- **id** (uuid, primary key)
- **risk_analysis_id** (uuid, foreign key to `risk_analyses.id`)
- **original_text** (text, the scanned snippet)
- **risk_status** (enum: `green`, `yellow`, `red`)
- **rationale** (text, AI explanation)
- **suggested_rewrite** (text, AI proposed change)
- **user_overridden_status** (enum: `green`, `yellow`, `red`, NULL)
- **user_override_rationale** (text, mandatory if status overridden)
- **is_gap** (boolean, true if AI identifies a missing mandatory clause)

### GoldenRule
Global rules defined by Admins that govern AI reasoning.

- **id** (uuid, primary key)
- **admin_id** (uuid, foreign key to `profiles.id`)
- **rule_text** (text, non-negotiable legal requirement)
- **priority** (int, highest takes precedence)
- **is_active** (boolean)

## Relationships

- **Document (1) <-> (*) RiskAnalysis**: A document can have multiple timestamped scan versions.
- **RiskAnalysis (1) <-> (*) ClauseAnalysis**: One scan produces multiple clause assessments.
- **Admin (1) <-> (*) GoldenRule**: Admins manage global rules that impact all AI analysis.

## State Transitions

- **Document Status**: `uploaded` -> `scanning` -> `analyzed` -> `reviewed`
- **RiskAnalysis Status**: `pending` -> `completed` OR `failed`
