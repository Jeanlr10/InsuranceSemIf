"""One triage run: guard the policy and the claim, score every clause, and build the output and the log entry.

Output contract: it never contains claim text. Rationales name only words from the policy text and keys of the
fixed lexicon. The four decision words never appear in any output; a final scan fails closed if one does.

With a model scorer (I2), the model's P(A) is the score. The lexical scorer still runs: it gives the rationale, and
a clause where the two disagree by MARGIN or more is listed for reading by hand (a tripwire, not a vote).
"""
import json
import re
import uuid
from dataclasses import dataclass

from . import VERSION, code_sha256
from .guard import CLAIM_GROUPS, POLICY_GROUPS, POLICY_MAX_CHARS, inspect, normalize, notes, sha256
from .scorer import NAME, PARAMS, Scorer, lexicon_sha256
from .scorer import VERSION as SCORER_VERSION
from .segment import segment
from .text import DECISION_WORDS

THRESHOLD = 0.5
MARGIN = 0.6   # placeholder until calibrated on the I2 train split (ix.calibrate)
NOTICE = ("TRIAGE ONLY. This ranks policy clauses by likely relevance to a claim. It makes no coverage decision. "
          "A person reads the clauses and decides.")
CLAIM_ID = re.compile(r"[A-Za-z0-9_.-]{1,40}")
DECISION = re.compile(r"\b(" + "|".join(DECISION_WORDS) + r")\b", re.I)
NOT_SCORED = "not scored"


class OutputContractError(RuntimeError):
    pass


def check_claim_id(claim_id: str) -> str:
    if not isinstance(claim_id, str) or not CLAIM_ID.fullmatch(claim_id):
        raise ValueError("claim_id must be 1 to 40 characters from A-Z a-z 0-9 _ . -")
    return claim_id


def band(score: float, threshold: float) -> str:
    if score >= threshold:
        return "likely relevant"
    if score >= threshold / 2:
        return "possibly relevant"
    return "unlikely relevant"


def scrub(text: str) -> str:
    """Policy-derived strings (headings) may contain decision words; they are masked, never echoed."""
    return DECISION.sub("[term]", text)


def rationale(cs, max_terms: int = 4) -> str:
    parts = []
    for t in cs.terms:
        if DECISION.search(t.surface) or (t.via and DECISION.search(t.via)):
            continue
        word = f'"{t.surface}"' + (" (in heading)" if t.in_heading else "")
        parts.append(f"{word} via {t.via}" if t.via else word)
        if len(parts) == max_terms:
            break
    if not parts:
        return "No shared or related terms."
    return "Shares " + ", ".join(parts) + "."


@dataclass
class Run:
    output: dict      # what the user sees (also hashed into the log)
    entry: dict       # the log entry body, before seq, ts and hashes
    policy_text: str
    claim_text: str


def triage(policy_text: str, claim_text: str, claim_id: str, *, threshold: float = THRESHOLD,
           defense: bool = True, run_id: str | None = None, model=None) -> Run:
    check_claim_id(claim_id)
    run_id = run_id or uuid.uuid4().hex

    # Policy: normalize, split, and check every clause with the strict pattern set. A flagged clause fails closed.
    pnorm = normalize(policy_text, POLICY_MAX_CHARS)
    clauses = segment(pnorm.text)
    suspect_clauses = []
    for c in clauses:
        ins = inspect(f"{c.heading}. {c.body}", POLICY_GROUPS, POLICY_MAX_CHARS)
        if ins.flags:
            suspect_clauses.append(c.id)

    # Claim: full guard. With defense off, flagged sentences are still detected and reported, but scored.
    cins = inspect(claim_text, CLAIM_GROUPS)
    scored_text = cins.kept_text if defense else cins.all_text
    scores = Scorer(clauses, skip=set(suspect_clauses)).score(scored_text)
    lexical = {k: round(v.score, 4) for k, v in scores.items()}
    disagree = []
    if model is not None:
        mp = model.p_yes(scored_text, [c for c in clauses if c.id not in suspect_clauses])
        for k, v in mp.items():
            scores[k].score = v
            if abs(round(v, 4) - lexical[k]) >= MARGIN:
                disagree.append(k)

    rows = []
    for c in clauses:
        if c.id in suspect_clauses:
            rows.append((0.0, c, NOT_SCORED, "Not scored: this clause matched instruction patterns, so its text was "
                                             "kept out of scoring. Read it directly."))
        else:
            cs = scores[c.id]
            s = round(cs.score, 4)
            rows.append((s, c, band(s, threshold), rationale(cs)))
    order = sorted(range(len(rows)), key=lambda i: (rows[i][2] == NOT_SCORED, -rows[i][0], i))
    results = [{"rank": r, "clause_id": rows[i][1].id, "ref": rows[i][1].ref, "heading": scrub(rows[i][1].heading),
                "score": rows[i][0], "label": rows[i][2], "rationale": rows[i][3]} for r, i in enumerate(order, 1)]

    integrity = "suspect" if cins.suspect else "clean"
    claim_notes = notes(cins, defense)
    if suspect_clauses:
        claim_notes.append(f"Policy clauses {', '.join(suspect_clauses)} matched instruction patterns and were not "
                           f"scored.")
    if disagree:
        claim_notes.append(f"The model and the lexical check disagree on {', '.join(sorted(disagree))}; read these "
                           f"clauses by hand.")
    output = {
        "notice": NOTICE,
        "claim_id": claim_id,
        "threshold": threshold,
        "defense": "on" if defense else "off",
        "input_integrity": integrity,
        "claim_flags": cins.flags,
        "claim_notes": claim_notes,
        "policy_suspect_clauses": suspect_clauses,
        "scorer": model.NAME if model is not None else NAME,
        "scorer_disagreement": sorted(disagree),
        "results": results,
    }
    text = json.dumps(output, ensure_ascii=False)
    if DECISION.search(text):
        raise OutputContractError("output would contain a decision word; refusing to emit it")

    entry = {
        "run_id": run_id,
        "claim_id": claim_id,
        "model": {"name": NAME, "version": SCORER_VERSION, "package_version": VERSION, "parameters": PARAMS,
                  "lexicon_sha256": lexicon_sha256(), "code_sha256": code_sha256(),
                  **({"neural": {"name": model.NAME, "gguf_sha256": model.sha256, "margin": MARGIN}}
                     if model is not None else {})},
        "params": {"threshold": threshold, "defense": defense},
        "inputs": {"policy_sha256": sha256(policy_text), "claim_sha256": sha256(claim_text),
                   "scored_claim_sha256": sha256(scored_text)},
        "integrity": {
            "input_integrity": integrity,
            "claim_flags": cins.flags,
            "claim_notes": claim_notes,
            "claim_chars_removed": dict(cins.norm.removed),
            "claim_controls_replaced": cins.norm.controls_replaced,
            "claim_mixed_script": cins.norm.mixed_script,
            "claim_truncated": cins.norm.truncated,
            "policy_suspect_clauses": suspect_clauses,
            "policy_chars_removed": dict(pnorm.removed),
        },
        "scores": {r["clause_id"]: r["score"] for r in sorted(results, key=lambda r: r["clause_id"])},
        **({"lexical_scores": lexical} if model is not None else {}),
        "above_threshold": [r["clause_id"] for r in results if r["label"] == "likely relevant"],
        "output_sha256": output_sha256(output),
    }
    return Run(output, entry, policy_text, claim_text)


def output_sha256(output: dict) -> str:
    return sha256(json.dumps(output, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
