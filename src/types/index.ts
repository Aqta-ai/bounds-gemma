// Minimal types for the Gemma 4 PHI redaction toolkit. Mirrors the
// upstream Bounds Pro production schema so that consumers can drop
// either implementation in without changing call sites.

export type PiiType =
  | 'PERSON'
  | 'ADDRESS'
  | 'EMAIL'
  | 'PHONE'
  | 'IBAN'
  | 'SSN'
  | 'PASSPORT'
  | 'ID_NUMBER'
  | 'DATE_OF_BIRTH'
  | 'IP_ADDRESS'
  | 'CREDIT_CARD'
  | 'URL'
  | 'ORG'
  | 'MISC'
  | 'HEALTH_DATA'
  | 'CONFIDENTIAL'
  | 'PROPRIETARY'
  | 'LEGAL_CLAUSE'

export type DetectionSource =
  | 'NER'
  | 'REGEX'
  | 'OCR'
  | 'MANUAL'
  | 'FACE'
  | 'GEMMA'

export interface BBox {
  x: number
  y: number
  width: number
  height: number
}

/**
 * The shape consumers of this toolkit emit and aggregate. Mirrors the
 * production redaction schema. Gemma-produced detections set
 * `source: 'GEMMA'`, `enabled: false` (reviewer must opt in per item),
 * and `ruleId` to one of the six healthcare PHI category ids the
 * worker is constrained to.
 */
export interface Detection {
  id: string
  type: PiiType
  text: string
  /** Replacement token used during PDF redaction, e.g. "[HEALTH_DATA_001]". */
  token: string
  pageIndex: number
  boundingBox: BBox
  /** 0 to 1; healthcare detections enforce a confidence floor of 0.75. */
  confidence: number
  source: DetectionSource
  /** UI default: false for Gemma; reviewer opts in per item. */
  enabled: boolean
  ruleId?: string
  /** One-sentence human-readable justification surfaced in the review UI. */
  reason?: string
}
