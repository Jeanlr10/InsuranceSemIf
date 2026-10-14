"""One adversarial round: GLM 5.3 Flash attacks the current I2 checkpoint, with feedback.

For each target (a training claim and a clause to promote or demote):
  stage A: Flash writes 3 attacks; each is measured against the current checkpoint.
  stage B: Flash sees its own attempts with the measured probability before and after, plus a few attacks that
           worked on other targets, and writes 3 more.
Kept for training: every attack that moved its target by more than 0.05 toward its goal, plus 20% of the rest so
the pool keeps covering what the model already resists.

    python -m ix.loop --round 1 --adapter runs/i2_r1/adapter --targets 600
"""
import argparse
import json
import random

from . import DATA, RUNS
from .attacks import FEEDBACK, TRAIN_FAMILIES, load_corpus, measure, pick_targets, summarize, write_attacks
from .llm import FLASH, Budget, pmap
from .model import load


def rows_text(rs: list[dict]) -> str:
    return "\n".join(f"- \"{x['payload']}\": P {x['p_clean']:.2f} -> {x['p_attacked']:.2f} "
                     f"({'worked' if x['success'] else 'failed'})" for x in rs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--targets", type=int, default=600)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = random.Random(1000 + args.round * 7 + args.seed)
    pols = {p["pid"]: p for p in load_corpus("train")}
    model, tok = load(args.adapter)
    budget = Budget(args.budget)
    jobs = pick_targets(list(pols.values()), args.targets, TRAIN_FAMILIES, r)

    def gen(job, feedback=""):
        return write_attacks(job, pols, FLASH, TRAIN_FAMILIES, budget, feedback=feedback)

    stage_a = pmap(gen, jobs, workers=16)
    flat_a = measure(model, tok, [a for rs in stage_a if rs for a in rs], pols)
    by_job = {}
    for a in flat_a:
        by_job.setdefault((a["pid"], a["cid"], a["target_ref"], a["goal"], a["family"]), []).append(a)
    wins = [a for a in flat_a if a["success"]]

    def gen_b(job):
        mine = by_job.get((job["pid"], job["cid"], job["target_ref"], job["goal"], job["family"]), [])
        if not mine:
            return []
        others = [w for w in wins if w["goal"] == job["goal"] and w["cid"] != job["cid"]]
        shown = mine + r.sample(others, min(3, len(others)))
        return gen(job, FEEDBACK.format(rows=rows_text(shown)))

    stage_b = pmap(gen_b, jobs, workers=16)
    flat_b = measure(model, tok, [a for rs in stage_b if rs for a in rs], pols)
    for a in flat_a:
        a["stage"] = "A"
    for a in flat_b:
        a["stage"] = "B"
    every = flat_a + flat_b
    keep = [a for a in every if a["shift"] > 0.05 or r.random() < 0.2]
    name = f"attacks_train_r{args.round}.jsonl"
    with (DATA / name).open("w") as f:
        for i, a in enumerate(keep):
            f.write(json.dumps({"aid": f"train-r{args.round}-{i:05d}", "round": args.round, **a},
                               ensure_ascii=False) + "\n")
    report = {"round": args.round, "against": args.adapter, "stage_A": summarize(flat_a),
              "stage_B": summarize(flat_b), "kept": len(keep), "calls": budget.calls,
              "cost": round(budget.spent, 4)}
    (RUNS / f"loop_r{args.round}.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v["ALL"] if isinstance(v, dict) and "ALL" in v else v) for k, v in report.items()}))


if __name__ == "__main__":
    main()
