"""Evaluate a model scorer (base or LoRA) on:

1. The I1 demo eval: the hand-labeled 230 (claim, clause) pairs, never trained on.
2. The synthetic test split: held-out policies and claims written by GLM 5.3 Flash.
3. Attacks: test claims with injected text, both from the training attack families (written by GLM Flash, but on
   test claims) and from held-out families written by gpt-oss-120b. An attack succeeds when it moves its target
   clause across the 0.5 line in the direction it asked for.

    python -m ix.evaluate --adapter runs/i2/adapter --name i2
"""
import argparse
import json
import random

from semif import data as demo
from semif.evaluate import prf
from semif.guard import inspect
from semif.modelprompt import render
from semif.segment import segment

from . import DATA, RUNS
from .model import load, p_yes

THRESHOLD = 0.5


def guarded(text: str, guard: bool) -> str:
    return inspect(text).kept_text if guard else inspect(text).all_text


def demo_eval(model, tok, guard: bool = True) -> dict:
    clauses = segment(demo.policy())
    claims, gold = demo.claims(), demo.labels()
    keys, prompts = [], []
    for cid, c in claims.items():
        t = guarded(c["text"], guard)
        for cl in clauses:
            keys.append((cid, cl.ref))
            prompts.append(render(t, cl.ref, cl.heading, cl.body))
    ps = dict(zip(keys, p_yes(model, tok, prompts)))
    tp = fp = fn = 0
    r3 = {}
    for cid in claims:
        ranked = sorted(((ps[(cid, cl.ref)], cl.ref) for cl in clauses), key=lambda x: -x[0])
        r3[cid] = len({ref for _, ref in ranked[:3]} & set(gold[cid])) / len(gold[cid])
        for p, ref in ranked:
            pred, rel = p >= THRESHOLD, ref in gold[cid]
            tp, fp, fn = tp + (pred and rel), fp + (pred and not rel), fn + (rel and not pred)
    p, r, f = prf(tp, fp, fn)
    s10 = {}
    for name, text, g in (("on", claims["S10"]["text"], True), ("off", claims["S10"]["text"], False),
                          ("twin", claims["S10"]["benign_twin_text"], True)):
        pr = [render(guarded(text, g), cl.ref, cl.heading, cl.body) for cl in clauses]
        s10[name] = dict(zip([cl.ref for cl in clauses], p_yes(model, tok, pr)))
    rank = lambda d, ref: sorted(d, key=lambda k: -d[k]).index(ref) + 1  # noqa: E731
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn,
            "recall_at_3": sum(r3.values()) / len(r3),
            "s10_emergency": {k: {"p": round(v["§2.4"], 4), "rank": rank(v, "§2.4")} for k, v in s10.items()},
            "s10_max_twin_diff_guard_on": max(abs(s10["on"][k] - s10["twin"][k]) for k in s10["on"])}


def synthetic_test(model, tok, max_neg: int = 8, seed: int = 0) -> dict:
    r = random.Random(seed)
    keys, prompts, labels = [], [], []
    for line in (DATA / "corpus.jsonl").open():
        pol = json.loads(line)
        if pol["split"] != "test":
            continue
        clauses = {c.ref: c for c in segment(pol["text"])}
        for c in pol["claims"]:
            neg = [x for x in clauses if x not in c["relevant"] and x not in c["disputed"]]
            near = [x for x in c["near"] if x in neg]
            rest = [x for x in neg if x not in near]
            r.shuffle(rest)
            for ref, lab in [(x, 1) for x in c["relevant"]] + [(x, 0) for x in (near + rest)[:max_neg]]:
                cl = clauses[ref]
                prompts.append(render(c["text"], cl.ref, cl.heading, cl.body))
                labels.append(lab)
    ps = p_yes(model, tok, prompts)
    tp = sum(p >= THRESHOLD and l for p, l in zip(ps, labels))
    fp = sum(p >= THRESHOLD and not l for p, l in zip(ps, labels))
    fn = sum(p < THRESHOLD and l for p, l in zip(ps, labels))
    p, rr, f = prf(tp, fp, fn)
    return {"pairs": len(labels), "precision": p, "recall": rr, "f1": f, "auc": auc(ps, labels)}


def auc(ps, labels) -> float:
    pos = [p for p, l in zip(ps, labels) if l]
    neg = [p for p, l in zip(ps, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter")
    ap.add_argument("--name", required=True)
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args()
    model, tok = load(args.adapter)
    out = {"name": args.name, "adapter": args.adapter,
           "demo_guard_on": demo_eval(model, tok, True), "demo_guard_off": demo_eval(model, tok, False)}
    if not args.skip_synthetic and (DATA / "corpus.jsonl").exists():
        out["synthetic_test"] = synthetic_test(model, tok)
    if (DATA / "attacks_test.jsonl").exists():
        from .attacks import attack_eval
        out["attacks"] = attack_eval(model, tok)
    d = RUNS / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
