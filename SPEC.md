# InsuranceSemIf: design

## Intent

A "semantic if" over policy text. Given one policy and one claim description, it splits the policy into clauses,
scores each clause from 0 to 1 for how likely it governs the claim, writes one line of rationale per clause, and
returns a ranked list plus a JSON audit trail.

It triages text. It never decides coverage. A person makes the call.

The security design is the point of the project: every input is untrusted, nothing leaves the machine, and every
run can be audited and rebuilt.

## Rules

1. **Synthetic data only.** No real health information anywhere. The demo is safe to share in public, and a test
   scans the data files for personal-data patterns.
2. **Input is data, never instructions.** Policy text and claim text are both untrusted.
3. **Local only.** The reference build makes no network calls (a test runs it with sockets blocked).
4. **Append-only audit log** with enough in it to rebuild any flag.
5. **Output language.** The tool says "likely relevant", "possibly relevant", "unlikely relevant" or "not scored".
   It never uses "covered", "denied", "approved" or "excluded" as its own words. A final check refuses to emit any
   output that contains one.

## Components and trust boundaries

```
             untrusted                         |                trusted code                 |   records
                                               |                                             |
  policy text ──► normalize ──► segment ──► strict guard ──┐                                 |
                                               |   (flagged clause: fail closed, not scored) |
                                               |                           ▼                 |
  claim text  ──► normalize ──► full guard ──► quarantine ──► lexical scorer ──► output ─────┼──► audit log (hashes only)
                                               |   (flagged sentences: hash only)   contract |
                                               |                                             └──► input store (raw text)
```

| Part | File | Job |
|---|---|---|
| Segmenter | `semif/segment.py` | Splits on `§N.N Heading` lines into `C01`…`C23`, each with ref, heading and body. Drops text before the first clause. Falls back to blank-line paragraphs. |
| Guard | `semif/guard.py` | Normalizes, detects instruction sentences, quarantines them. |
| Scorer | `semif/scorer.py`, `semif/lexicon.json` | Deterministic BM25 with a small lexicon. No generative model. |
| Output contract | `semif/triage.py` | Ranked results, bands, rationale, decision-word check. |
| Audit | `semif/audit.py`, `semif/replay.py` | Hash-chained log, input store, `verify`, `replay`. |

## Input guard

Runs on every claim, and a stricter, smaller version runs on every policy clause.

1. **Normalize.** Unicode NFKC (so full-width letters fold to ASCII). Remove zero-width characters, bidirectional
   controls and all other invisible format characters, including Unicode tag characters used for "ASCII
   smuggling". Replace other control characters with spaces. Cap the length (5,000 characters for a claim). Flag
   words that mix Latin with Cyrillic or Greek letters.
2. **Detect.** Split into sentences on `.`, `!` or `?` followed by whitespace, and on newlines, so "1.0" stays
   whole. Test each sentence against five pattern groups, on a copy with Cyrillic and Greek look-alike letters
   folded to Latin:

   | Group | Catches |
   |---|---|
   | `override` | ignore / disregard / forget / bypass … instructions / rules / prompt |
   | `role` | "you are now", "act as", "pretend to be", "admin mode", "developer mode" |
   | `marker` | `system:`, `<system>`, `[INST]`, `<|im_start|>`-style chat-template tokens |
   | `score_manipulation` | assign / rank / output / set … relevance / score / clause / JSON |
   | `decision_demand` | state / confirm / mark … covered / approved / payable; "approve this claim" |

3. **Quarantine.** Drop every flagged sentence from the text that is scored. Keep its SHA-256, length and offsets
   for the log. The raw sentence is never written to the output or the log.
4. **Report.** The claim is marked `input_integrity: suspect`, with the flagged sentence indexes and a note that
   they were treated as data and never followed. Removed hidden characters and mixed-script words also mark the
   claim suspect.

**Policy text** uses only `override`, `role` and `marker`. Policies say "pay" and "covered" constantly, so the
other two groups would raise false alarms. A flagged clause **fails closed**: its text is not indexed or scored,
it is labeled "not scored", listed last, and named in `policy_suspect_clauses` so a person reads it directly.

**Why quarantine when the scorer cannot follow instructions?** Instruction text carries topic words. S10's
injection says "emergency", "ambulance" and "room", and those words move a word-overlap score: with the defense off,
§2.4 Emergency Services jumps from 0.00 to 0.81 and rank 2. Quarantine removes that lever.

## Scorer

Deterministic, so the injection defense is easy to prove and a log entry can be replayed exactly.

- Lower-case, join hyphenated words (`x-ray` → `xray`), keep letters only, drop stop words and claim boilerplate
  ("claim", "member", "plan", payment words). The four decision words are stop words too, so they can never move a
  score or appear in a rationale.
- A small suffix-stripping stemmer (services/service, imaging/image, therapies/therapy).
- Each claim word counts once, however often it repeats, which limits keyword stuffing.
- BM25 over the clauses (k1 1.2, b 0.75), with heading words counted twice.
- A hand-written lexicon of 65 entries maps claim words to related policy words at weight 0.6 ("IVF" →
  "fertility", "MRI" → "imaging", "gastrectomy" → "bariatric").
- Squash: `score = 1 - exp(-raw / 6)`. Fixed before the eval was run and not tuned after.
- Rationale: up to four terms by contribution, marking heading hits and lexicon links. Rationales only ever print
  words from the policy text or keys of the fixed lexicon, never free claim text.
- Bands: at or above the threshold (0.5) is "likely relevant", at or above half of it "possibly relevant", below
  that "unlikely relevant".

## Output contract

Every run prints a notice: triage only, no coverage decision, a person decides.

JSON fields: `notice`, `claim_id`, `threshold`, `defense`, `input_integrity`, `claim_flags`, `claim_notes`,
`policy_suspect_clauses`, `results[]` (`rank`, `clause_id`, `ref`, `heading`, `score`, `label`, `rationale`) and
`audit` (`run_id`, `seq`, `entry_hash`).

- `claim_id` must match `[A-Za-z0-9_.-]{1,40}`, so it cannot inject into the log or a terminal.
- Claim text is never echoed. Flags carry hashes, lengths and offsets only.
- Headings come from the policy. A decision word in a heading is masked as `[term]`.
- The serialized output is scanned for decision words before it is returned; a hit raises instead of emitting.

## Audit log

One JSON line per triage run in `audit/audit.jsonl`.

| Field | Content |
|---|---|
| `seq`, `ts`, `prev_hash`, `entry_hash` | The chain. `entry_hash` is SHA-256 of the canonical entry without that field. |
| `run_id`, `claim_id` | Run ID and pseudonymous claim ID. |
| `model` | Scorer name and version, parameters, lexicon SHA-256, code SHA-256. |
| `params` | Threshold, defense on or off. |
| `inputs` | SHA-256 of the policy, the claim, and the scored (cleaned) claim. |
| `integrity` | Flags (patterns, hashes, lengths, offsets), notes, removed characters by code point, mixed-script words (hashed), suspect policy clauses. |
| `scores` | Score for every clause. |
| `above_threshold` | Clause IDs at or above the threshold. |
| `output_sha256` | Hash of the output shown to the user. |

**Writing.** Open in append mode, take an exclusive lock, re-check the last entry's own hash (refuse to append if
it fails, or if the file does not end in a newline), write, flush, fsync.

**Inputs.** The log holds hashes only. The original texts go to a separate content-addressed store
(`audit/inputs/<sha256>.txt`, mode 600), so a flag can be rebuilt. In production this store is the sensitive part:
encrypt it, restrict access to it, and apply the retention policy to it.

**`verify`** walks the chain and reports edited entries, removed or reordered entries, and sequence gaps. It prints
the head hash. **`replay`** re-runs every logged run from the stored inputs and compares scores, flags and output
hash, and says whether the code or lexicon changed since the run.

**Limits.**
- The chain shows edits and deletions before the last entry. It does not show deletion of the last entries or of
  the whole file. Save the head hash somewhere else on a schedule and check it with `verify --head <hash>`.
- In production, let the OS or storage enforce append-only: `chattr +a` on the file, or object-lock (WORM) storage.
- Locking uses `fcntl`, so the reference build runs on Linux and macOS.

## Threat table

| Threat | Control | Test |
|---|---|---|
| Claim text tries to steer the scorer | Detect and quarantine instruction sentences. The scorer has no instruction-following path. | S10 equals its benign twin; §2.4 0.00 with defense on vs 0.81 off |
| Hidden characters hide an instruction | NFKC, strip zero-width, bidi and tag characters, flag mixed scripts, fold look-alikes for detection | Zero-width "ignore", Cyrillic "ignоre", full-width text, bidi and tag characters |
| Poisoned policy clause | Strict patterns on every clause, fail closed | Poisoned-policy test |
| Keyword stuffing | Each claim word counts once; BM25 saturation | Stuffed-claim test |
| Decision words planted to leak into output | Decision words are stop words; masked in headings; final scan fails closed | Decision-word scan of every output; forced-failure test |
| Log tampering | Hash chain, tail check before append, `verify`, saved head, `replay` | Edit, middle-delete, tail-delete-with-head, tampered-tail-append tests |
| Stored input swapped | Store files are checked against their hash on read | Altered-input test |
| Log injection | Claim ID pattern; no raw text in the log | ID validation test; log holds no claim text |
| Data leaves the machine | No network calls in the reference build | Triage and eval with sockets blocked |
| Tool read as a decision maker | Notice on every run, relevance bands, decision-word filter | Decision-word scan |
| Paraphrased injection with no trigger words | Not fully stopped by patterns. Limited by design: such text can only share words with a clause. | Known limit |

## If you swap in a hosted model

Keep the same input guard and output contract, then add:

- Wrap claim and clause text in fixed delimiters and tell the model the text inside is data.
- Give the model no tools and no network access.
- Force a JSON schema for the output, reject anything else, and clamp scores to 0 to 1.
- Keep running the lexical scorer. If the two disagree by more than a set margin on a clause, flag the claim for
  review. This is a cheap tripwire against a model that has been steered.
- Log the model name, the exact model version string, the prompt hash and the raw response hash.

**Tradeoff.** Document content leaves the machine. For real claims that needs a business associate agreement, a
check of the provider's data retention, and the insurer's sign-off. Hosted models also change under you, so
`replay` may stop matching: pin a version and log it. A model run on the insurer's own hardware avoids the first
problem but not the second.

## Known limits

- One person wrote the spec for the data, the labels and the lexicon, and the lexicon was written with the claims in
  view. The eval is a smoke test, not an estimate for real policies.
- The scorer is word overlap. It does not read negation or exceptions.
- A clause that matters only through another clause (prior authorization required by a benefit clause) is missed
  when the claim does not use its words.
- The 0 to 1 value is a squashed word-overlap score, not a calibrated probability.
