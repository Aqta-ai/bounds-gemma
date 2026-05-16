#!/usr/bin/env node
/**
 * Synthetic PHI training-dataset generator for bounds-gemma fine-tune.
 *
 * Uses local Ollama with a teacher model (gemma4:latest = e4b, ~9.6 GB)
 * to generate diverse clinical-note paragraphs along with the JSON
 * array of contextual PHI spans that the student fine-tune target
 * (gemma4:e2b) should learn to emit.
 *
 * Output: JSONL at finetune/data/phi_training.jsonl, one example per
 * line. Each example is an OpenAI-style messages array compatible with
 * Unsloth's SFTTrainer:
 *
 *   {
 *     "messages": [
 *       { "role": "system",    "content": "<the production system prompt>" },
 *       { "role": "user",      "content": "<clinical note paragraph>" },
 *       { "role": "assistant", "content": "<expected JSON array>" }
 *     ]
 *   }
 *
 * Why distillation: a teacher 4x the parameter count of the student
 * provides higher-quality labels than the student could produce alone,
 * giving the student a measurable lift after LoRA fine-tuning. Same
 * pattern used by Alpaca, Orca, and Phi-1.5.
 *
 * Privacy: every clinical note is synthetic. No real patient data is
 * used or generated. Names, dates, conditions, and locations are all
 * model-invented. Verify before using in any downstream evaluation.
 *
 * Usage (assumes Ollama running on localhost:11435):
 *   OLLAMA_HOST=http://127.0.0.1:11435 node finetune/scripts/generate_phi_dataset.mjs
 *
 * Default target: 500 examples. Set EXAMPLES_TARGET=1000 to expand.
 * Resume support: re-running picks up where the previous run stopped.
 */

import { writeFileSync, appendFileSync, existsSync, readFileSync } from 'node:fs'

const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://127.0.0.1:11435'
const TEACHER_MODEL = process.env.TEACHER_MODEL || 'gemma4:latest'
const EXAMPLES_TARGET = parseInt(process.env.EXAMPLES_TARGET || '500', 10)
const OUTPUT_PATH = 'finetune/data/phi_training.jsonl'

// The production system prompt the student model must learn to obey.
// Identical to bounds-gemma/src/workers/gemma.worker.ts SYSTEM_PROMPT,
// kept verbatim so the fine-tune trains for exact production conditions.
const PRODUCTION_SYSTEM_PROMPT = `You are a healthcare privacy auditor. The following is text extracted from a PDF page that may contain protected health information (PHI). Your job is to flag spans that contain PHI a regex or named-entity recogniser would MISS, focusing on these six categories ONLY:

  1. inline_diagnosis        Inline diagnosis without a structured label (e.g. "presents with generalised anxiety disorder").
  2. medication_mention      Medication name in running prose, not after a "Medications:" label (e.g. "she is on lithium").
  3. treatment_procedure     Treatment, procedure, or surgery mentioned in narrative (e.g. "underwent CABG last April").
  4. indirect_health_context Indirect references that imply health condition (e.g. "my therapist", "my insulin pump").
  5. sensitive_social        Sensitive social-category data (sexuality, gender identity, religion).
  6. genetic_reference       Genetic test results or family history (e.g. "BRCA1-positive", "father had Huntington's").

For each span you flag, return ONE entry in a JSON array:

  {
    "text":       "<the exact span as it appears in the input, byte-identical>",
    "type":       "HEALTH_DATA",
    "confidence": <0.0 to 1.0>,
    "ruleId":     "gemma:<one of the six category ids above>",
    "reason":     "<one short sentence explaining why this span is PHI a reviewer should consider>"
  }

Rules:
- Return ONLY a JSON array. No prose, no markdown, no preamble.
- If no PHI is present, return [].
- Confidence MUST be at least 0.75. Below that, omit the span.
- The text field MUST appear verbatim in the input. Do NOT paraphrase, expand, or correct.
- DO NOT flag patient names, dates, addresses, MRNs, phone numbers, emails, or other structured PHI; the regex and NER layers handle those. Your job is contextual content only.
- If you are unsure, leave the span out. False positives in healthcare are worse than false negatives at this stage.`

// Templates for diverse synthetic clinical notes. The teacher gets a
// scenario seed + a list of PHI categories to include. The teacher
// invents the note text AND produces the labeled JSON output in one
// pass. We then verify the labels are in-corpus before keeping.
const SCENARIOS = [
  'a primary-care follow-up visit for a chronic condition',
  'a psychiatry intake note with medication history',
  'an emergency-department discharge summary',
  'a cardiology consult note',
  'an oncology treatment-plan summary',
  'a paediatric well-child visit with family history',
  'a post-surgical follow-up note',
  'a women\'s-health visit including genetic testing notes',
  'an endocrinology note for a patient with diabetes',
  'a sports-medicine return-to-play assessment',
  'a sleep-medicine consultation',
  'a gastroenterology follow-up after a procedure',
  'a neurology evaluation for headache and seizure history',
  'an ophthalmology visit with relevant family history',
  'a workplace occupational-health screening note',
  'a community-pharmacy medication-review consultation',
  'a hospice intake assessment',
  'a transplant pre-evaluation note',
  'a dermatology consultation with genetic markers',
  'a substance-use disorder treatment plan summary',
]

const CATEGORY_COMBINATIONS = [
  ['inline_diagnosis', 'medication_mention'],
  ['inline_diagnosis', 'treatment_procedure'],
  ['medication_mention', 'indirect_health_context'],
  ['treatment_procedure', 'genetic_reference'],
  ['inline_diagnosis', 'medication_mention', 'treatment_procedure'],
  ['indirect_health_context', 'sensitive_social'],
  ['genetic_reference', 'inline_diagnosis'],
  ['inline_diagnosis', 'medication_mention', 'genetic_reference', 'indirect_health_context'],
  ['medication_mention'],
  ['inline_diagnosis'],
  ['treatment_procedure'],
  ['indirect_health_context'],
  ['sensitive_social', 'inline_diagnosis'],
  ['genetic_reference'],
  // Negative example: no PHI categories — teacher should return []
  [],
]

const GENERATOR_PROMPT = (scenario, categories) => {
  const catGuide = {
    inline_diagnosis: 'an inline diagnosis without a structured label (e.g. "presents with generalised anxiety disorder", "Type 2 Diabetes Mellitus")',
    medication_mention: 'a medication name in running prose (e.g. "she is on lithium", "started Adderall")',
    treatment_procedure: 'a treatment or procedure in narrative (e.g. "underwent CABG last April", "biopsy revealed")',
    indirect_health_context: 'an indirect reference implying a health condition (e.g. "my therapist", "her insulin pump")',
    sensitive_social: 'sensitive social-category data (e.g. "identifies as non-binary", "patient is HIV-positive")',
    genetic_reference: 'a genetic test result or family history (e.g. "BRCA1-positive", "father had Huntington\'s")',
  }

  const requireCats = categories.length > 0
  const catInstructions = requireCats
    ? `Include ALL of these PHI categories in the note (one example each, at minimum):\n${categories.map((c) => `  - ${c}: ${catGuide[c]}`).join('\n')}`
    : 'Do NOT include any of the six contextual PHI categories. The note must contain a patient name, date, and address only (structured PHI handled by other layers).'

  return `Output ONLY a JSON object with two fields: "note" and "detections".

Scenario: ${scenario}.

${catInstructions}

The note must also contain a patient name AND a date (these are structured PHI handled by other layers; do NOT include them in detections).

For each contextual PHI span you include in the note, add an entry to "detections" with these fields:
  - text: the exact span as it appears in the note, BYTE-IDENTICAL (do not paraphrase)
  - type: "HEALTH_DATA"
  - confidence: a number between 0.85 and 1.00
  - ruleId: "gemma:" followed by the category name (e.g. "gemma:inline_diagnosis")
  - reason: a short sentence explaining the flag

If categories list above is empty, "detections" MUST be an empty array [].

Example output format (for reference, not for copying):
{
  "note": "Mr. John Doe was seen today, 2024-05-15, for follow-up. His Type 2 Diabetes Mellitus remains poorly controlled. He continues on metformin 1000mg twice daily.",
  "detections": [
    { "text": "Type 2 Diabetes Mellitus", "type": "HEALTH_DATA", "confidence": 0.95, "ruleId": "gemma:inline_diagnosis", "reason": "Inline diagnosis without a structured label." },
    { "text": "metformin", "type": "HEALTH_DATA", "confidence": 0.92, "ruleId": "gemma:medication_mention", "reason": "Medication mentioned in running prose." }
  ]
}

Output the JSON object now. Start with {.`
}

async function callOllama(prompt, model = TEACHER_MODEL) {
  const res = await fetch(`${OLLAMA_HOST}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model,
      messages: [{ role: 'user', content: prompt }],
      stream: false,
      think: false,
      options: { temperature: 0.85, num_predict: 2048 },
      format: 'json',
    }),
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`Ollama ${res.status}: ${body.slice(0, 200)}`)
  }
  const data = await res.json()
  return data.message?.content ?? data.response ?? ''
}

function stripFences(raw) {
  return raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/i, '').trim()
}

// The six PHI categories we train the student to emit. Anything else
// (e.g. gemma:symptom_mention, gemma:family_history) is silently
// dropped — those drift from the production schema and would teach
// the student to emit categories that the parser later rejects.
const ALLOWED_RULE_IDS = new Set([
  'gemma:inline_diagnosis',
  'gemma:medication_mention',
  'gemma:treatment_procedure',
  'gemma:indirect_health_context',
  'gemma:sensitive_social',
  'gemma:genetic_reference',
])

function validateExample(parsed) {
  if (typeof parsed !== 'object' || parsed === null) return null
  const note = typeof parsed.note === 'string' ? parsed.note.trim() : ''
  const detections = Array.isArray(parsed.detections) ? parsed.detections : null
  if (!note || note.length < 40) return null
  if (detections === null) return null

  const noteNfc = note.normalize('NFC')
  const cleanedDetections = []
  for (const d of detections) {
    if (typeof d !== 'object' || d === null) continue
    const text = typeof d.text === 'string' ? d.text.trim() : ''
    if (!text) continue
    const textNfc = text.normalize('NFC')
    if (!noteNfc.includes(textNfc)) continue
    const confidence = typeof d.confidence === 'number' ? d.confidence : 0
    if (confidence < 0.75) continue
    const ruleId = typeof d.ruleId === 'string' ? d.ruleId : ''
    if (!ALLOWED_RULE_IDS.has(ruleId)) continue
    const reason = typeof d.reason === 'string' ? d.reason.trim() : ''
    if (!reason) continue
    cleanedDetections.push({
      text,
      type: 'HEALTH_DATA',
      confidence: Math.min(1, Math.max(0.75, confidence)),
      ruleId,
      reason,
    })
  }

  return { note, detections: cleanedDetections }
}

function alreadyGenerated() {
  if (!existsSync(OUTPUT_PATH)) return 0
  const content = readFileSync(OUTPUT_PATH, 'utf8')
  return content.split('\n').filter((l) => l.trim().length > 0).length
}

function pickRandom(arr) {
  return arr[Math.floor(Math.random() * arr.length)]
}

async function main() {
  const already = alreadyGenerated()
  console.log(`Target: ${EXAMPLES_TARGET} examples. Already have: ${already}. Generating ${Math.max(0, EXAMPLES_TARGET - already)} more.`)
  console.log(`Teacher: ${TEACHER_MODEL} via ${OLLAMA_HOST}`)
  console.log(`Output:  ${OUTPUT_PATH}`)

  let count = already
  let attempts = 0
  let failures = 0
  const t0 = Date.now()

  while (count < EXAMPLES_TARGET) {
    attempts++
    const scenario = pickRandom(SCENARIOS)
    const categories = pickRandom(CATEGORY_COMBINATIONS)
    const prompt = GENERATOR_PROMPT(scenario, categories)

    try {
      const raw = await callOllama(prompt)
      const parsed = JSON.parse(stripFences(raw))
      const validated = validateExample(parsed)
      if (!validated) {
        failures++
        if (failures % 10 === 0) {
          console.warn(`[${count}/${EXAMPLES_TARGET}] ${failures} bad outputs so far`)
        }
        continue
      }

      const example = {
        messages: [
          { role: 'system', content: PRODUCTION_SYSTEM_PROMPT },
          { role: 'user', content: validated.note },
          { role: 'assistant', content: JSON.stringify(validated.detections, null, 2) },
        ],
      }
      appendFileSync(OUTPUT_PATH, JSON.stringify(example) + '\n')
      count++

      if (count % 25 === 0 || count === EXAMPLES_TARGET) {
        const elapsed = ((Date.now() - t0) / 1000).toFixed(0)
        const rate = (count / (Date.now() - t0) * 1000 * 60).toFixed(1)
        console.log(`[${count}/${EXAMPLES_TARGET}] ${elapsed}s elapsed, ~${rate} examples/min, ${failures} drops`)
      }
    } catch (err) {
      failures++
      console.warn(`[${count}/${EXAMPLES_TARGET}] error: ${String(err).slice(0, 100)}`)
      if (failures > attempts * 0.5 && attempts > 20) {
        console.error('Too many failures. Check teacher model output. Aborting.')
        process.exit(1)
      }
    }
  }

  console.log(`Done. ${count} examples in ${OUTPUT_PATH}. ${failures} drops in ${attempts} attempts.`)
}

main().catch((err) => {
  console.error(err)
  process.exit(1)
})
