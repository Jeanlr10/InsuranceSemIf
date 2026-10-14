"""Eval over every (claim, clause) pair: 10 claims x 23 clauses = 230 pairs. Writes EVAL.md.

The threshold (0.5) and every scorer constant were fixed before this was first run. The sweep is shown instead of
picking a better threshold afterwards."""
from . import ROOT, data
from .scorer import PARAMS
from .triage import THRESHOLD, triage

SWEEP = [round(0.30 + 0.05 * i, 2) for i in range(9)]


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def counts(runs: dict, gold: dict, thr: float, ids=None) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for cid, out in runs.items():
        if ids and cid not in ids:
            continue
        for r in out["results"]:
            pred, rel = r["score"] >= thr, r["ref"] in gold[cid]
            tp += pred and rel
            fp += pred and not rel
            fn += rel and not pred
    return tp, fp, fn


def run() -> dict:
    pol, claims, gold = data.policy(), data.claims(), data.labels()
    runs = {cid: triage(pol, c["text"], cid).output for cid, c in claims.items()}
    n_pairs = sum(len(o["results"]) for o in runs.values())
    tp, fp, fn = counts(runs, gold, THRESHOLD)
    p, r, f = prf(tp, fp, fn)
    r3 = {cid: len({x["ref"] for x in o["results"][:3]} & set(gold[cid])) / len(gold[cid]) for cid, o in runs.items()}
    types = {}
    for cid, c in claims.items():
        types.setdefault(c["type"], []).append(cid)
    by_type = {t: prf(*counts(runs, gold, THRESHOLD, set(ids))) + (len(ids),) for t, ids in types.items()}
    sweep = [(t,) + prf(*counts(runs, gold, t)) + counts(runs, gold, t) for t in SWEEP]

    s10 = claims["S10"]
    on = runs["S10"]
    off = triage(pol, s10["text"], "S10", defense=False).output
    twin = triage(pol, s10["benign_twin_text"], "S10").output

    def at(out, ref):
        return next(x for x in out["results"] if x["ref"] == ref)

    inj = {
        "flags": [f["sentence_index"] for f in on["claim_flags"]],
        "patterns": sorted({p for f in on["claim_flags"] for p in f["patterns"]}),
        "on": at(on, "§2.4"), "off": at(off, "§2.4"),
        "twin_equal": {x["ref"]: x["score"] for x in on["results"]} == {x["ref"]: x["score"] for x in twin["results"]},
        "top_on": [x["ref"] for x in on["results"][:3]], "top_off": [x["ref"] for x in off["results"][:3]],
        "clean_flags": {cid: len(o["claim_flags"]) for cid, o in runs.items() if cid != "S10"},
        "policy_suspect": on["policy_suspect_clauses"],
    }
    return {"pairs": n_pairs, "tp": tp, "fp": fp, "fn": fn, "p": p, "r": r, "f": f, "recall3": r3,
            "by_type": by_type, "sweep": sweep, "runs": runs, "gold": gold, "injection": inj}


def report(m: dict) -> str:
    L = []
    w = L.append
    w("# Eval\n")
    w(f"Stated before the first run: **threshold {THRESHOLD}**, squash `score = 1 - exp(-raw / {PARAMS['squash_k']:g})`, "
      f"BM25 k1 {PARAMS['k1']}, b {PARAMS['b']}, heading words x{PARAMS['heading_weight']}, lexicon weight "
      f"{PARAMS['lexicon_weight']}. None of these were changed after seeing results.\n")
    w(f"Unit: one (claim, clause) pair, {m['pairs']} pairs. A pair counts as flagged when its score is at or above "
      f"the threshold (\"likely relevant\"). Regenerate with `python -m semif eval`.\n")
    w("## At the stated threshold\n")
    w("| Precision | Recall | F1 | True pos. | False pos. | Missed | Mean recall@3 |")
    w("|---|---|---|---|---|---|---|")
    mean3 = sum(m["recall3"].values()) / len(m["recall3"])
    w(f"| {m['p']:.2f} | {m['r']:.2f} | {m['f']:.2f} | {m['tp']} | {m['fp']} | {m['fn']} | {mean3:.2f} |\n")
    w("Many false positives and few or no misses is the intended failure mode for triage: a reviewer reads an extra "
      "clause rather than missing one.\n")
    w("## By claim type\n")
    w("| Type | Claims | Precision | Recall | F1 |")
    w("|---|---|---|---|---|")
    for t, (p, r, f, n) in m["by_type"].items():
        w(f"| {t} | {n} | {p:.2f} | {r:.2f} | {f:.2f} |")
    w("\n## Threshold sweep (shown, not used to pick the threshold)\n")
    w("| Threshold | Precision | Recall | F1 | True pos. | False pos. | Missed |")
    w("|---|---|---|---|---|---|---|")
    for t, p, r, f, tp, fp, fn in m["sweep"]:
        w(f"| {t:.2f}{' (stated)' if t == THRESHOLD else ''} | {p:.2f} | {r:.2f} | {f:.2f} | {tp} | {fp} | {fn} |")
    w("\n## Per claim\n")
    w("| Claim | Hand labels | Top 3 | Recall@3 | Likely relevant | Missed at threshold |")
    w("|---|---|---|---|---|---|")
    for cid, o in m["runs"].items():
        g = m["gold"][cid]
        top = ", ".join(x["ref"] for x in o["results"][:3])
        likely = [x["ref"] for x in o["results"] if x["score"] >= THRESHOLD]
        missed = [ref for ref in g if ref not in likely]
        w(f"| {cid} | {', '.join(g)} | {top} | {m['recall3'][cid]:.2f} | {len(likely)} | {', '.join(missed) or '-'} |")
    i = m["injection"]
    w("\n## Injection test (S10)\n")
    w("S10 is three sentences about a cosmetic chin implant followed by five injected sentences that demand §2.4 "
      "Emergency Services be ranked first and the claim be called payable. The scorer cannot follow instructions; "
      "the risk is that the injected words (\"emergency\", \"ambulance\", \"room\") move a word-overlap score.\n")
    w(f"- Sentences flagged: {', '.join(map(str, i['flags']))} ({len(i['flags'])} of 5 injected), patterns: "
      f"{', '.join(i['patterns'])}.")
    w("\n| Defense | §2.4 score | §2.4 rank | §2.4 label | Top 3 |")
    w("|---|---|---|---|---|")
    w(f"| on | {i['on']['score']:.2f} | {i['on']['rank']} | {i['on']['label']} | {', '.join(i['top_on'])} |")
    w(f"| off | {i['off']['score']:.2f} | {i['off']['rank']} | {i['off']['label']} | {', '.join(i['top_off'])} |\n")
    w(f"- With the defense on, every clause score equals the benign twin's (S10 without the five sentences): "
      f"**{'yes' if i['twin_equal'] else 'NO'}**.")
    w(f"- The §2.4 score of {i['on']['score']:.2f} with the defense on means no word overlap after quarantine. It "
      "does not mean §2.4 is irrelevant in general.")
    w(f"- False flags on the clean claims S01 to S09: {sum(i['clean_flags'].values())}. Policy clauses flagged: "
      f"{len(i['policy_suspect'])}.\n")
    w("## Limits\n")
    w("- One person wrote the labels and the lexicon, and the lexicon was written with these claims in view. The "
      "policy and claims were generated by GLM 5.3 Flash from a spec written by the same person. These numbers are a "
      "smoke test, not an estimate for real policies.")
    w(f"- {m['pairs']} pairs is small. One label change moves precision or recall by several points.")
    w("- The scorer is word overlap. It does not read negation or exceptions: \"not paid\" and \"paid\" look the same "
      "to it.")
    w("- The 0 to 1 value is a squashed word-overlap score, not a calibrated probability.")
    w("- A clause that matters only through another clause is missed when the claim does not use its words: a "
      "benefit clause that requires prior authorization does not pull in §3.1 by itself. Following the policy's own "
      "§ cross-references is the obvious next step. It was not added after seeing these results, so the misses stay "
      "in the table.")
    w("- Pattern detection does not stop a paraphrased injection with no trigger words. That text is scored, but "
      "the only thing it can do is share words with a clause.")
    return "\n".join(L) + "\n"


def main(write: bool = True) -> dict:
    m = run()
    text = report(m)
    if write:
        (ROOT / "EVAL.md").write_text(text, encoding="utf-8")
    return m
