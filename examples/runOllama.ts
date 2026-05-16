/**
 * End-to-end smoke test of the bounds-gemma pipeline against a local
 * Ollama daemon. Reads a sample clinical-note paragraph, sends it to
 * Gemma 4 E2B with the production system prompt, parses + validates
 * the response, and prints the contextual-PHI detections that survived
 * the in-corpus + confidence-floor checks.
 *
 * Usage:
 *   ollama pull gemma4:e2b
 *   ollama serve &
 *   npm run example:ollama
 *
 * This script does not connect to a Bounds Pro server, does not ship
 * the text anywhere off-device, and does not require an API key. The
 * only network call is the localhost POST to Ollama.
 */

import { parseAndValidate, HEALTHCARE_CONFIDENCE_FLOOR } from '../src/workers/gemmaParse'

const SAMPLE = `Patient is a 42 year old who presents with generalised anxiety disorder. She has been on lithium for three years and has been taking Adderall for ADHD since her teens. She underwent CABG last April and follow-up cardiology is at City General. Family history is significant for Huntington's disease on her father's side. She mentioned her therapist on Tuesday and her insulin pump alarm woke her at 3am.`

const SYSTEM_PROMPT = `You are a healthcare privacy auditor. The following is text extracted from a PDF page that may contain protected health information (PHI). Your job is to flag spans that contain PHI a regex or named-entity recogniser would MISS, focusing on these six categories ONLY:

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
- Confidence MUST be at least ${HEALTHCARE_CONFIDENCE_FLOOR}. Below that, omit the span.
- The text field MUST appear verbatim in the input. Do NOT paraphrase, expand, or correct.
- DO NOT flag patient names, dates, addresses, MRNs, phone numbers, emails, or other structured PHI; the regex and NER layers handle those. Your job is contextual content only.
- If you are unsure, leave the span out. False positives in healthcare are worse than false negatives at this stage.`

interface OllamaChatResponse {
  message?: { content?: string }
  response?: string
  error?: string
}

async function main(): Promise<void> {
  const url = 'http://localhost:11434/api/chat'

  process.stdout.write('Posting sample clinical note to Ollama (gemma4:e2b)...\n')
  const t0 = Date.now()

  let res: Response
  try {
    res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'gemma4:e2b',
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          { role: 'user', content: SAMPLE },
        ],
        stream: false,
        options: {
          temperature: 0.1,
          num_predict: 600,
        },
      }),
    })
  } catch (err) {
    process.stderr.write(
      `Could not reach Ollama at ${url}. Is the daemon running? (\`ollama serve\` in another terminal)\n`,
    )
    process.stderr.write(`Underlying error: ${String(err)}\n`)
    process.exit(1)
  }

  if (!res.ok) {
    process.stderr.write(`Ollama returned HTTP ${res.status}.\n`)
    process.stderr.write(await res.text())
    process.stderr.write('\nIs the model pulled? Try: ollama pull gemma4:e2b\n')
    process.exit(1)
  }

  const data = (await res.json()) as OllamaChatResponse
  const elapsed = Date.now() - t0
  const raw = data.message?.content ?? data.response ?? ''
  process.stdout.write(`Ollama returned in ${elapsed}ms.\n\n`)
  process.stdout.write(`Raw model output:\n${raw}\n\n`)

  const detections = parseAndValidate(raw, SAMPLE)
  process.stdout.write(`After parseAndValidate (in-corpus + confidence floor ${HEALTHCARE_CONFIDENCE_FLOOR}):\n`)
  process.stdout.write(`${detections.length} detection${detections.length === 1 ? '' : 's'} survived.\n\n`)
  for (const d of detections) {
    process.stdout.write(`  • [${d.ruleId}] (conf ${d.confidence.toFixed(2)}) "${d.text}"\n`)
    process.stdout.write(`    Reason: ${d.reason}\n\n`)
  }
}

void main()
