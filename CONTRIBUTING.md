# Contributing

Bug reports and pull requests are welcome. A few ground rules to keep the toolkit healthy.

## Before opening a PR

1. **Run the parser tests.** `npm test` exercises 16 invariants of the JSON validator and the in-corpus check. PRs that break a parser test almost always have a deeper problem.
2. **Keep the system prompt narrow.** The six HIPAA Safe Harbor #17 categories are deliberately scoped to what regex and NER cannot do. Adding a seventh category is a non-trivial review; please open an issue first.
3. **Confidence floor changes need a justification.** The 0.75 floor in `gemmaParse.ts` was tuned for healthcare and informed by the rate of false positives on a 500-page in-house clinical-notes corpus. Lowering it for a non-clinical use case is reasonable; document the corpus you tuned against.
4. **Never weaken the in-corpus check.** A Gemma-emitted span that is not byte-identical (under NFC) to a span in the input is a hallucination by definition. Dropping such spans silently is the contract; do not relax it.

## What is out of scope

- Structured PHI (phone numbers, SSNs, MRNs, addresses). Those are regex / NER work; this toolkit is contextual only.
- PDF bounding-box resolution. The consumer maps detection text back to PDF coordinates; this toolkit is text-in, text-spans-out.
- Auditor pairing with a second model. The Bounds Pro production pipeline pairs Gemma 4 26B paraphraser with Gemma 4 31B auditor; that lives in the closed-source upstream. This toolkit is the on-device, audit-free path.

## Reporting a security issue

Email the maintainer; do not open a public issue with a working exploit. Privacy-claim regressions (any path where document text could leak off-device) are treated as security issues.

## Licence and contribution

By submitting a pull request you agree to license your contribution under the same Apache-2.0 terms as the rest of the toolkit.
