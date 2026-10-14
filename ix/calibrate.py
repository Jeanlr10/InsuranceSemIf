"""Set the tripwire margin (semif.triage.MARGIN) on the TRAIN split, never the test split.

For every clean train pair, compute |model P(A) - lexical score|. The margin is the quantile of that gap at which the
tripwire fires on --rate of pairs (default 5%). Also reports how many of the model's own errors the tripwire catches
at that margin, which is the number that matters to a reviewer.

    python -m ix.calibrate --adapter runs/i2_r3/adapter
"""
import argparse
import json

from semif.modelprompt import render
from semif.scorer import Scorer

from . import RUNS
from .attacks import load_corpus
from .model import load, p_yes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--rate", type=float, default=0.05)
    args = ap.parse_args()
    model, tok = load(args.adapter)
    prompts, rows = [], []
    for p in load_corpus("train"):
        clauses = list(p["clauses"].values())
        sc = Scorer(clauses)
        for c in p["claims"]:
            lex = sc.score(c["text"])
            for cl in clauses:
                if cl.ref in c["disputed"]:
                    continue
                prompts.append(render(c["text"], cl.ref, cl.heading, cl.body))
                rows.append((lex[cl.id].score, int(cl.ref in c["relevant"])))
    ps = p_yes(model, tok, prompts)
    gaps = sorted(abs(m - l) for m, (l, _) in zip(ps, rows))
    margin = round(gaps[int((1 - args.rate) * (len(gaps) - 1))], 3)
    wrong = [(m, l) for m, (l, y) in zip(ps, rows) if (m >= 0.5) != bool(y)]
    caught = sum(abs(m - l) >= margin for m, l in wrong)
    res = {"adapter": args.adapter, "pairs": len(rows), "rate": args.rate, "margin": margin,
           "fires_on": round(sum(g >= margin for g in gaps) / len(gaps), 4),
           "model_errors": len(wrong), "errors_caught": caught,
           "errors_caught_rate": round(caught / len(wrong), 4) if wrong else None}
    (RUNS / "calibrate.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
