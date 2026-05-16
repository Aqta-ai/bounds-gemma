# bounds-gemma

**On-device Gemma 4 contextual PHI redaction.** A small, focused TypeScript toolkit that uses Google's Gemma 4 to catch the protected-health-information shapes that regex and named-entity recognition systematically miss: inline diagnoses in clinical prose, medication mentions, treatment narratives, indirect health context, sensitive social data, and genetic references. Runs in a browser via WebLLM, locally via Ollama, or in any environment that can speak HTTP to a Gemma 4 endpoint. Never on a server.

Live demo: <https://bounds.pro>

This is the open-source pipeline that powers the contextual-PHI layer of Bounds Pro, a closed-source PDF redaction workspace. The toolkit on its own is enough to reproduce that layer end to end on your own documents.

---

## Why this exists

The HIPAA Safe Harbor de-identification standard at 45 CFR 164.514(b)(2) lists eighteen identifier categories. The first sixteen are structured — phone numbers, social-security numbers, medical-record numbers, dates of birth — and the long-standing rule-based redactors handle them. Identifier #17 is *"any other unique identifying number, characteristic, or code"*, and the surrounding clinical narrative is where it lives: a sentence that names a diagnosis without a label, a paragraph that mentions a medication in passing, an aside about a "therapist" or "insulin pump" that re-identifies the patient when triangulated with the rest of the document.

Existing PDF redaction tools force a choice no healthcare reviewer should have to make: send the document to a cloud API and trust their privacy posture, or use a regex-only desktop tool that demonstrably misses everything contextual. This toolkit's argument is that a small, capable on-device model — Gemma 4 E2B at int4 quantisation, ~1.5 GB on disk, running on the user's own browser via WebGPU — closes the gap without ever shipping document bytes off-device.

## What it does

`bounds-gemma` exports a small surface area centred on a single async call:

```ts
import { startGemmaJob, getGemmaBackend } from 'bounds-gemma/pipeline/GemmaWorker'

// Probe which backend is reachable (Ollama localhost first, WebLLM fallback,
// unavailable if neither works). Cached after first probe.
const backend = await getGemmaBackend()

// Run a page's extracted text through Gemma. Returns the contextual-PHI
// detections the regex and NER layers would have missed.
const detections = await startGemmaJob({
  text: pageText,
  pageIndex: 0,
})
```

Each detection is a `{ text, type, confidence, ruleId, reason }` object. Confidence has a healthcare-only floor of 0.75; below that, the detection is silently dropped. The text field is verified to be a byte-identical substring of the input page (with NFC Unicode normalisation), so model hallucinations and paraphrases never reach the consumer.

## Architecture

```
Input page text
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ GemmaWorker (main-thread facade)                                │
│  ├─ probes Ollama at localhost:11434/api/tags                   │
│  ├─ falls back to WebLLM via @mlc-ai/web-llm in a Worker        │
│  └─ chunks long pages, dispatches one model call per chunk      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ Gemma 4 inference                                               │
│  ├─ Ollama:   model=gemma4:e2b                                  │
│  ├─ WebLLM:   model=gemma-4-E2B-it-q4f16_1-MLC                  │
│  ├─ system prompt: six HIPAA Safe Harbor #17 categories         │
│  └─ output contract: JSON array, no prose                       │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ gemmaParse (validator)                                          │
│  ├─ strips fences and prose preamble                            │
│  ├─ enforces JSON schema and confidence floor (0.75)            │
│  ├─ verifies every detection text is in-corpus (NFC compared)   │
│  └─ rejects hallucinations silently                             │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
Detection[] for downstream PDF redaction
```

Three guardrails make this safe for healthcare paraphrase tasks:

1. **In-corpus verification.** Every Gemma-emitted span must be a byte-identical substring of the input page text after Unicode NFC normalisation. Model hallucinations and paraphrases are dropped silently before they ever reach the review surface.
2. **Confidence floor of 0.75.** Tuned specifically for healthcare; below it, candidates are omitted. This is a single constant in `gemmaParse.ts` and easy to lower for non-clinical use cases.
3. **Default-off in the consumer UI.** Every Gemma detection arrives with `enabled: false`. The downstream reviewer must opt in per item. Surface-level acceptance is never automatic.

## Two execution paths

### Ollama (preferred for production)

```bash
ollama pull gemma4:e2b
ollama serve
```

Then point any consumer at `http://localhost:11434/api/chat`. The toolkit probes this URL at start-up; if reachable, it routes all subsequent calls there. Sub-second latency per chunk on consumer hardware, zero model-CDN traffic, fully offline after the model pull.

### WebLLM (no-install, in-browser)

Serve your application under cross-origin isolation:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

The toolkit dynamic-imports `@mlc-ai/web-llm` on first use and loads `gemma-4-E2B-it-q4f16_1-MLC` from the MLC CDN. Weights are cached in browser storage; subsequent sessions are fully offline. First load is slow (~1.5 GB download); subsequent loads are instant.

## Install

```bash
npm install bounds-gemma
# Optional, only if you want the WebLLM browser path:
npm install @mlc-ai/web-llm
```

The package has no required runtime dependencies. `@mlc-ai/web-llm` is a peer dependency you only pull in if you use the browser fallback. Ollama is a separate install (`brew install ollama` or equivalent).

## Run the example

```bash
git clone https://github.com/Aqta-ai/bounds-gemma.git
cd bounds-gemma
npm install
ollama pull gemma4:e2b
ollama serve &  # in another terminal
npm run example:ollama
```

The example runs a sample clinical-note paragraph through Gemma 4 and prints the contextual-PHI detections.

## Run the tests

```bash
npm install
npm test
```

16 unit tests in `src/__tests__/gemmaParse.test.ts` cover the parser, validator, in-corpus check, NFC normalisation, fence-stripping, malformed-JSON handling, and confidence-floor enforcement. They run in <1 second with no model required.

## What this toolkit deliberately does NOT do

- It does not handle structured PHI (phone numbers, SSNs, dates, MRNs, addresses). Those are the regex and NER layers' job; combine this toolkit with [a regex PII detector](https://github.com/microsoft/presidio) for full Safe Harbor coverage.
- It does not draw bounding boxes on PDFs. That is the consumer's job; the toolkit returns text spans and lets the consumer resolve them to PDF coordinates.
- It does not run an auditor. The cross-check pattern in the closed-source Bounds Pro pairs Gemma 4 26B with Gemma 4 31B as paraphraser plus auditor; the on-device toolkit ships only the paraphraser side and relies on verbatim-wins-ties as the safety floor.
- It does not phone home. No analytics, no telemetry, no model-CDN ping. Verify with the Network tab.

## Source attribution

Built and maintained by Anya Chueayen ([@anyaparanya](https://github.com/anyaparanya)). Released under Apache-2.0 (see `LICENSE`). The Gemma family and the Gemma Prohibited Use Policy are governed by their own terms; using this toolkit means you accept Google's terms for Gemma 4 as well. The HIPAA Safe Harbor identifier list is in the public domain.

## Acknowledgements

- [Google DeepMind](https://deepmind.google/) for Gemma 4 and the open weights.
- [MLC LLM](https://llm.mlc.ai/) and [WebLLM](https://github.com/mlc-ai/web-llm) for the browser runtime.
- [Ollama](https://ollama.com/) for the local-first inference daemon.
- The [Centers for Medicare and Medicaid Services](https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html) for the public-domain HIPAA Safe Harbor specification.
