export type SuggestedDataset = {
  id: string
  name: string
  description?: string
  kind?: 'pdf' | 'csv' | 'markdown' | 'images' | 'json' | 'timeseries'
  size?: string
}

export type SampleProject = {
  id: string
  slug: string
  title: string
  description: string
  updatedAt: string // ISO date
  downloadSize: string // e.g., "2.2GB"
  dataSize: string // e.g., "560MB"
  primaryModel?: string
  models?: string[]
  tags?: string[]
  samplePrompt?: string
  embeddingStrategy?: string
  retrievalStrategy?: string
  datasetCount?: number
  suggestedDatasets?: SuggestedDataset[]
}

export const sampleProjects: SampleProject[] = [
  {
    id: 'aircraft-mx',
    slug: 'aircraft-mx',
    title: 'Aircraft Maintenance Assistant (Airgapped)',
    description:
      'Diagnoses faults and guides airmen through step-by-step repairs using offline manuals, checklists, flight logs, and parts cross-references. Built for disconnected, secure environments.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '2.2GB',
    dataSize: '130MB',
    primaryModel: 'TinyLlama',
    models: ['TinyLlama'],
    tags: ['maintenance', 'fleet', 'ops'],
    samplePrompt:
      'You are an experienced aircraft maintenance technician with 15+ years of experience working on military and commercial aircraft.',
    embeddingStrategy: 'PDF Simple',
    retrievalStrategy: 'Hybrid BM25 + embedding',
    datasetCount: 5,
    suggestedDatasets: [
      {
        id: 'flight-logs',
        name: 'Flight Logs',
        description: 'Recent sorties, squawks, and maintenance write-ups (CSV)',
        kind: 'csv',
        size: '74MB',
      },
      {
        id: 'mx-manuals',
        name: 'Maintenance Manuals & T.O.s',
        description:
          'Aircraft-specific technical orders and maintenance guides (PDF)',
        kind: 'pdf',
        size: '1.1GB',
      },
      {
        id: 'fault-codes',
        name: 'Fault Codes & MEL/CDL',
        description:
          'Common fault codes, MEL/CDL references, limits (Markdown)',
        kind: 'markdown',
        size: '8MB',
      },
      {
        id: 'parts-catalog',
        name: 'Illustrated Parts Catalog',
        description: 'Exploded diagrams and part numbers for assemblies (PDF)',
        kind: 'pdf',
        size: '420MB',
      },
      {
        id: 'checklists',
        name: 'Checklists & SOPs',
        description:
          'Pre-/post-flight, maintenance and safety checklists (Markdown)',
        kind: 'markdown',
        size: '5MB',
      },
    ],
  },
  {
    id: 'med-records',
    slug: 'medical-records',
    title: 'Healthcare Records Summarizer & Triage',
    description:
      'Ingests EHR notes, labs, and imaging reports to produce clinician summaries, problem lists, and follow-up suggestions with PHI-safe guardrails.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '1.1GB',
    dataSize: '420MB',
    primaryModel: 'TinyLlama',
    models: ['TinyLlama'],
    tags: ['healthcare', 'EHR', 'PHI'],
    samplePrompt:
      'Summarize the patient record, list active problems, and suggest next steps. Use clinical tone and avoid speculation.',
    embeddingStrategy: 'Section-aware clinical note splitter',
    retrievalStrategy: 'Hybrid BM25 + embedding with medical synonyms',
    datasetCount: 5,
    suggestedDatasets: [
      {
        id: 'clinical-notes',
        name: 'Clinical Notes',
        description: 'Progress notes, H&P, discharge summaries (PDF/Markdown)',
        kind: 'pdf',
        size: '210MB',
      },
      {
        id: 'labs-vitals',
        name: 'Labs & Vitals',
        description: 'Lab panels and vital signs history (CSV)',
        kind: 'csv',
        size: '38MB',
      },
      {
        id: 'radiology-reports',
        name: 'Radiology Reports',
        description: 'Imaging narratives and impressions (PDF)',
        kind: 'pdf',
        size: '95MB',
      },
      {
        id: 'meds-allergies',
        name: 'Medications & Allergies (FHIR)',
        description: 'Medication list, allergies, and reactions (JSON)',
        kind: 'json',
        size: '12MB',
      },
      {
        id: 'care-guidelines',
        name: 'Care Guidelines',
        description: 'Clinical guidelines and pathways (PDF)',
        kind: 'pdf',
        size: '160MB',
      },
    ],
  },
  {
    id: 'case-law-2025',
    slug: 'case-law-2025',
    title: 'Case Law Research Copilot (2025)',
    description:
      'Searches recent opinions, extracts holdings, compares jurisdictions, and drafts argument outlines with citation tracking.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '2.8GB',
    dataSize: '600MB',
    primaryModel: 'TinyLlama',
    models: ['TinyLlama'],
    tags: ['legal', 'research'],
    samplePrompt:
      'Given an issue, gather relevant cases, summarize holdings, and provide a short argument outline with citations.',
    embeddingStrategy: 'Case-aware chunking with headings and citations',
    retrievalStrategy: 'Hybrid BM25 + embedding with citation boosting',
    datasetCount: 5,
    suggestedDatasets: [
      {
        id: 'recent-opinions',
        name: 'Recent Opinions',
        description: 'Federal and state appellate opinions (PDF/HTML)',
        kind: 'pdf',
        size: '320MB',
      },
      {
        id: 'statutes-regs',
        name: 'Statutes & Regulations',
        description: 'Relevant codes and regulations (PDF/HTML)',
        kind: 'pdf',
        size: '240MB',
      },
      {
        id: 'briefs-work-product',
        name: 'Firm Briefs & Memos',
        description: 'Internal work product and templates (Markdown/PDF)',
        kind: 'markdown',
        size: '46MB',
      },
      {
        id: 'citation-graph',
        name: 'Citation Graph',
        description:
          'Shepardization signals and citation relationships (CSV/JSON)',
        kind: 'csv',
        size: '18MB',
      },
      {
        id: 'dockets',
        name: 'Dockets & Filings',
        description: 'Docket entries and pleadings (PDF/HTML)',
        kind: 'pdf',
        size: '210MB',
      },
    ],
  },
  {
    id: 'cust-support-copilot',
    slug: 'customer-support-copilot',
    title: 'Customer Support Copilot for SaaS',
    description:
      'Retrieves KB articles, past tickets, and changelogs to draft replies, suggest troubleshooting steps, and auto-route complex cases.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '850MB',
    dataSize: '260MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['support', 'SaaS'],
    samplePrompt:
      'Draft a concise, empathetic response with steps pulled from the most relevant KB entry and similar resolved tickets.',
    embeddingStrategy: 'FAQ/KB aware markdown splitter',
    retrievalStrategy: 'Hybrid with recency boost',
    datasetCount: 3,
    suggestedDatasets: [
      {
        id: 'kb',
        name: 'Knowledge Base',
        description: 'KB articles and FAQs (Markdown)',
        kind: 'markdown',
        size: '62MB',
      },
      {
        id: 'tickets',
        name: 'Resolved Tickets',
        description: 'Historical tickets and resolutions (CSV)',
        kind: 'csv',
        size: '85MB',
      },
      {
        id: 'changelogs',
        name: 'Changelogs & Release Notes',
        description: 'Product changes by version (Markdown)',
        kind: 'markdown',
        size: '12MB',
      },
    ],
  },
  {
    id: 'sales-rfp-assistant',
    slug: 'sales-rfp',
    title: 'Sales RFP & Proposal Assistant',
    description:
      'Parses RFPs, maps requirements to product capabilities, drafts compliant responses, and flags gaps/risks.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '1.2GB',
    dataSize: '340MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['sales', 'rfp'],
    samplePrompt:
      'Create a compliant response section with cited capabilities and call out any unmet requirements.',
    embeddingStrategy: 'Section-title and table-aware chunking',
    retrievalStrategy: 'Hybrid with requirement-term synonyms',
    datasetCount: 4,
    suggestedDatasets: [
      {
        id: 'rfps',
        name: 'RFP Documents',
        description: 'Customer RFPs (PDF/Word)',
        kind: 'pdf',
      },
      {
        id: 'product-capabilities',
        name: 'Product Capabilities',
        description: 'Feature matrices and limits (Markdown)',
        kind: 'markdown',
      },
      {
        id: 'security',
        name: 'Security & Compliance Docs',
        description: 'Security whitepapers and controls (PDF)',
        kind: 'pdf',
      },
      {
        id: 'win-library',
        name: 'Win Library',
        description: 'Prior answers and boilerplate (Markdown)',
        kind: 'markdown',
      },
    ],
  },
  {
    id: 'financial-filings-analyst',
    slug: 'fin-filings-analyst',
    title: 'Financial Filings Analyst',
    description:
      'Extracts metrics from 10-Ks/10-Qs and earnings calls, builds comps tables, and generates variance commentary.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '2.0GB',
    dataSize: '700MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['finance'],
    samplePrompt:
      'Produce a concise comps table and summarize YoY variances with drivers.',
    embeddingStrategy: 'Table-aware financial chunking',
    retrievalStrategy: 'Hybrid with numeric proximity scoring',
    datasetCount: 4,
    suggestedDatasets: [
      {
        id: 'sec-10k-10q',
        name: '10-Ks & 10-Qs',
        description: 'SEC filings (PDF/HTML)',
        kind: 'pdf',
        size: '520MB',
      },
      {
        id: 'earnings-calls',
        name: 'Earnings Call Transcripts',
        description: 'Call transcripts (PDF/Markdown)',
        kind: 'pdf',
        size: '160MB',
      },
      {
        id: 'company-metadata',
        name: 'Company Metadata',
        description: 'Tickers, sectors, and peer lists (CSV)',
        kind: 'csv',
        size: '7MB',
      },
      {
        id: 'kpi-dictionary',
        name: 'KPI Dictionary',
        description: 'Definitions of metrics and formulas (Markdown)',
        kind: 'markdown',
        size: '4MB',
      },
    ],
  },
  {
    id: 'devops-runbook-guide',
    slug: 'devops-runbook',
    title: 'DevOps Runbook Guide',
    description:
      'Correlates logs/alerts, proposes likely incident causes, and walks engineers through validated runbook steps and rollback commands.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '650MB',
    dataSize: '220MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['devops', 'sre'],
    samplePrompt:
      'Given these alerts and logs, propose the most likely cause and a safe, auditable remediation sequence.',
    embeddingStrategy: 'Log line windowing with metadata',
    retrievalStrategy: 'BM25 with recency filters',
    datasetCount: 3,
    suggestedDatasets: [
      {
        id: 'alerts',
        name: 'Alerts',
        description: 'Alert payloads and metadata (JSON)',
        kind: 'json',
        size: '28MB',
      },
      {
        id: 'logs',
        name: 'Logs',
        description: 'Application and system logs (JSON)',
        kind: 'json',
        size: '130MB',
      },
      {
        id: 'runbooks-sre',
        name: 'Runbooks',
        description: 'Incident runbooks and rollback steps (Markdown)',
        kind: 'markdown',
        size: '9MB',
      },
    ],
  },
  {
    id: 'manufacturing-quality-coach',
    slug: 'mfg-quality',
    title: 'Manufacturing Quality Coach',
    description:
      'Classifies defects from inspection notes, surfaces tolerances/SOPs, and recommends corrective actions with traceability.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '900MB',
    dataSize: '310MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['manufacturing', 'quality'],
    samplePrompt:
      'Summarize likely root cause and propose corrective actions referencing relevant SOP sections.',
    embeddingStrategy: 'SOP-aware chunking with tables',
    retrievalStrategy: 'Hybrid with part-number boosting',
    datasetCount: 4,
    suggestedDatasets: [
      {
        id: 'inspection-notes',
        name: 'Inspection Notes',
        description: 'QA/inspection logs (CSV)',
        kind: 'csv',
        size: '52MB',
      },
      {
        id: 'sop',
        name: 'SOPs & Work Instructions',
        description: 'Standard procedures (Markdown/PDF)',
        kind: 'markdown',
        size: '26MB',
      },
      {
        id: 'tolerances',
        name: 'Specs & Tolerances',
        description: 'Engineering drawings/specs (PDF)',
        kind: 'pdf',
        size: '190MB',
      },
      {
        id: 'defect-library',
        name: 'Defect Library',
        description: 'Known defects and fixes (Markdown)',
        kind: 'markdown',
        size: '6MB',
      },
    ],
  },
  {
    id: 'hr-policy-assistant',
    slug: 'hr-policy',
    title: 'HR Policy & Document Assistant',
    description:
      'Answers policy questions, generates letters (PIPs, offers), and logs decisions for audit compliance.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '520MB',
    dataSize: '140MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['hr'],
    samplePrompt:
      'Draft a policy-consistent response and produce a letter template with placeholders.',
    embeddingStrategy: 'Section-aware policy chunking',
    retrievalStrategy: 'Hybrid with policy precedence rules',
    datasetCount: 3,
    suggestedDatasets: [
      {
        id: 'handbook',
        name: 'Employee Handbook',
        description: 'Policies and procedures (PDF/Markdown)',
        kind: 'pdf',
        size: '38MB',
      },
      {
        id: 'templates',
        name: 'Letter Templates',
        description: 'Offer/PIP/termination templates (Markdown)',
        kind: 'markdown',
        size: '3MB',
      },
      {
        id: 'local-regs',
        name: 'Local Regulations',
        description: 'Regional labor rules (PDF)',
        kind: 'pdf',
        size: '92MB',
      },
    ],
  },
  {
    id: 'compliance-controls-navigator',
    slug: 'compliance-controls',
    title: 'Compliance Controls Navigator (SOC 2/ISO 27001)',
    description:
      'Maps controls to evidence, drafts audit responses, and schedules renewal reminders.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '480MB',
    dataSize: '190MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['compliance', 'security'],
    samplePrompt:
      'Generate an auditor-ready response with linked evidence items and owners.',
    embeddingStrategy: 'Control/evidence pair-aware chunking',
    retrievalStrategy: 'Hybrid with control identifier boosting',
    datasetCount: 4,
    suggestedDatasets: [
      {
        id: 'controls',
        name: 'Control Catalog',
        description: 'SOC2/ISO control texts (Markdown)',
        kind: 'markdown',
        size: '11MB',
      },
      {
        id: 'evidence',
        name: 'Evidence Register',
        description: 'Evidence items and owners (CSV/JSON)',
        kind: 'csv',
        size: '5MB',
      },
      {
        id: 'policies',
        name: 'Internal Policies',
        description: 'Security and IT policies (PDF/Markdown)',
        kind: 'pdf',
        size: '73MB',
      },
      {
        id: 'audit-history',
        name: 'Audit History',
        description: 'Past findings and remediations (Markdown)',
        kind: 'markdown',
        size: '7MB',
      },
    ],
  },
  {
    id: 'real-estate-lease-analyzer',
    slug: 'lease-analyzer',
    title: 'Real Estate Lease Analyzer',
    description:
      'Extracts key terms (rent escalations, options, CAM) from leases and summarizes obligations/risk.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '780MB',
    dataSize: '260MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['real-estate'],
    samplePrompt:
      'Summarize economic terms, obligations, and unusual clauses with section citations.',
    embeddingStrategy: 'Clause-aware legal chunking',
    retrievalStrategy: 'Hybrid with clause-type boosting',
    datasetCount: 3,
    suggestedDatasets: [
      {
        id: 'leases',
        name: 'Lease Documents',
        description: 'Commercial leases (PDF)',
        kind: 'pdf',
        size: '260MB',
      },
      {
        id: 'amendments',
        name: 'Amendments & Addenda',
        description: 'Lease amendments (PDF)',
        kind: 'pdf',
        size: '90MB',
      },
      {
        id: 'term-dictionary',
        name: 'Term Dictionary',
        description: 'Common terms/definitions (Markdown)',
        kind: 'markdown',
        size: '2MB',
      },
    ],
  },
  {
    id: 'education-course-builder',
    slug: 'edu-course-builder',
    title: 'Education Course Builder & Tutor',
    description:
      'Generates lesson plans, quizzes, and explanations from curriculum outlines; adapts difficulty to learner progress.',
    updatedAt: '2025-08-15T00:00:00.000Z',
    downloadSize: '680MB',
    dataSize: '210MB',
    primaryModel: 'TinyLama',
    models: ['TinyLama'],
    tags: ['education'],
    samplePrompt:
      'Produce a lesson plan and two assessment questions aligned to the standard, with answer keys.',
    embeddingStrategy: 'Section-aware curriculum splitting',
    retrievalStrategy: 'BM25 + embedding with standard mapping',
    datasetCount: 3,
    suggestedDatasets: [
      {
        id: 'curriculum',
        name: 'Curriculum Standards',
        description: 'Standards/competencies (Markdown/PDF)',
        kind: 'markdown',
        size: '14MB',
      },
      {
        id: 'lesson-library',
        name: 'Lesson Library',
        description: 'Past lesson plans and activities (Markdown)',
        kind: 'markdown',
        size: '28MB',
      },
      {
        id: 'assessment-bank',
        name: 'Assessment Bank',
        description: 'Question bank and rubrics (CSV/Markdown)',
        kind: 'csv',
        size: '6MB',
      },
    ],
  },
]
