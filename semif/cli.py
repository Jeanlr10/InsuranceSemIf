"""Command line.

    python -m semif triage S10                 rank clauses for a demo claim (logged)
    python -m semif triage --file claim.txt --id C-77
    python -m semif eval                       230-pair eval and injection test, writes EVAL.md
    python -m semif verify                     walk the audit chain
    python -m semif replay                     re-run logged runs from stored inputs and compare
    python -m semif triage S10 --gguf i2.gguf  score with the I2 model (llama-server on CPU) instead of BM25
"""
import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from . import ROOT, audit, data
from .triage import NOTICE, THRESHOLD, triage

LOG = ROOT / "audit" / "audit.jsonl"
STORE = ROOT / "audit" / "inputs"


def scorer(a):
    """The I2 model scorer when --gguf (or SEMIF_GGUF) is given, else nothing: the lexical scorer is used."""
    gguf = a.gguf or os.environ.get("SEMIF_GGUF")
    if not gguf:
        return contextlib.nullcontext()
    from .modelscorer import ModelScorer
    return ModelScorer(gguf, threads=a.threads)


def cmd_triage(a) -> int:
    if a.file:
        if not a.id:
            sys.exit("--file needs --id")
        claim_id, text = a.id, Path(a.file).read_text(encoding="utf-8")
    else:
        claims = data.claims()
        if a.claim not in claims:
            sys.exit(f"unknown demo claim; choose one of {', '.join(claims)}")
        claim_id, text = a.claim, claims[a.claim]["text"]
    policy = Path(a.policy).read_text(encoding="utf-8") if a.policy else data.policy()
    try:
        with scorer(a) as model:
            run = triage(policy, text, claim_id, threshold=a.threshold, defense=a.defense == "on", model=model)
    except ValueError as e:
        sys.exit(str(e))
    out = run.output
    if not a.no_log:
        st = audit.Store(a.store)
        st.put(policy)
        st.put(text)
        entry = audit.append(a.log, run.entry)
        out["audit"] = {"run_id": entry["run_id"], "seq": entry["seq"], "entry_hash": entry["entry_hash"]}
    else:
        out["audit"] = {"logged": False}
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0
    print(NOTICE)
    print()
    print(f"Claim {out['claim_id']}  input integrity: {out['input_integrity'].upper()}  defense: {out['defense']}  "
          f"threshold: {out['threshold']}  scorer: {out['scorer']}")
    for n in out["claim_notes"]:
        print(f"  ! {n}")
    print()
    print(f"{'#':>2}  {'ref':<6} {'score':>5}  {'label':<18} heading")
    for r in out["results"][: a.top]:
        print(f"{r['rank']:>2}  {r['ref']:<6} {r['score']:>5.2f}  {r['label']:<18} {r['heading']}")
        print(f"{'':>34}{r['rationale']}")
    if len(out["results"]) > a.top:
        print(f"    ... {len(out['results']) - a.top} more clauses below (--top {len(out['results'])} to see all)")
    print()
    if out["audit"].get("seq"):
        print(f"Logged as seq {out['audit']['seq']}, run {out['audit']['run_id']}, entry {out['audit']['entry_hash'][:16]}")
    return 0


def cmd_eval(a) -> int:
    from .evaluate import main
    m = main(write=not a.no_write)
    i = m["injection"]
    print(f"{m['pairs']} pairs at threshold {THRESHOLD}: precision {m['p']:.2f}, recall {m['r']:.2f}, F1 {m['f']:.2f} "
          f"({m['tp']} TP, {m['fp']} FP, {m['fn']} missed); mean recall@3 "
          f"{sum(m['recall3'].values()) / len(m['recall3']):.2f}")
    print(f"S10 injection: {len(i['flags'])} sentences flagged; §2.4 score {i['on']['score']:.2f} (rank {i['on']['rank']}) "
          f"with defense on, {i['off']['score']:.2f} (rank {i['off']['rank']}) with it off; equals benign twin: "
          f"{'yes' if i['twin_equal'] else 'NO'}")
    if not a.no_write:
        print(f"Wrote {ROOT / 'EVAL.md'}")
    return 0


def cmd_verify(a) -> int:
    if not Path(a.log).exists():
        print(f"No log at {a.log} yet.")
        return 0
    ok, problems, last = audit.verify(a.log, a.head)
    for p in problems:
        print(f"FAIL {p}")
    if ok:
        n = last['seq'] if last else 0
        print(f"OK: {n} {'entry' if n == 1 else 'entries'}, chain intact.")
    if last:
        print(f"Head: seq {last['seq']} {last['entry_hash']}  (save this elsewhere; check later with --head)")
    return 0 if ok else 1


def cmd_replay(a) -> int:
    from .replay import replay
    try:
        with scorer(a) as model:
            rows = replay(a.log, a.store, a.run_id, model)
    except FileNotFoundError as e:
        sys.exit(f"missing: {e.filename}")
    bad = 0
    skipped = [r for r in rows if r.get("skipped")]
    rows = [r for r in rows if not r.get("skipped")]
    for r in skipped:
        print(f"seq {r['seq']:>4} {r['claim_id']:<10} SKIPPED  model-scored run; pass --gguf to replay it")
    for r in rows:
        ok = r["scores_match"] and r["output_match"] and r["flags_match"]
        bad += not ok
        why = [] if ok else [f"clauses differ: {', '.join(r['changed_clauses']) or 'output only'}"]
        if r["code_changed"]:
            why.append("code changed since this run")
        if r["weights_changed"]:
            why.append("model weights changed since this run")
        if r["lexicon_changed"]:
            why.append("lexicon changed since this run")
        print(f"seq {r['seq']:>4} {r['claim_id']:<10} {'MATCH' if ok else 'DIFFERS'}  {'; '.join(why)}")
    print(f"{len(rows) - bad} of {len(rows)} runs match.")
    return 0 if not bad else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m semif", description=NOTICE)
    ap.add_argument("--log", type=Path, default=LOG, help="audit log (default audit/audit.jsonl)")
    ap.add_argument("--store", type=Path, default=STORE, help="original-input store (default audit/inputs)")
    ap.add_argument("--gguf", help="I2 model GGUF; scores with the model instead of BM25 (or set SEMIF_GGUF)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for the model scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("triage", help="rank policy clauses for one claim")
    t.add_argument("claim", nargs="?", default="S10", help="demo claim ID, S01 to S10")
    t.add_argument("--file", help="claim text file instead of a demo claim")
    t.add_argument("--id", help="claim ID for --file (A-Z a-z 0-9 _ . -, up to 40)")
    t.add_argument("--policy", help="policy text file (default: the synthetic demo policy)")
    t.add_argument("--threshold", type=float, default=THRESHOLD)
    t.add_argument("--defense", choices=["on", "off"], default="on", help="off scores flagged sentences too")
    t.add_argument("--top", type=int, default=8)
    t.add_argument("--json", action="store_true")
    t.add_argument("--no-log", action="store_true")
    e = sub.add_parser("eval", help="230-pair eval and injection test; writes EVAL.md")
    e.add_argument("--no-write", action="store_true")
    v = sub.add_parser("verify", help="walk the audit chain")
    v.add_argument("--head", help="a head hash saved earlier; fails if it is gone")
    r = sub.add_parser("replay", help="re-run logged runs and compare")
    r.add_argument("run_id", nargs="?")
    a = ap.parse_args(argv)
    return {"triage": cmd_triage, "eval": cmd_eval, "verify": cmd_verify, "replay": cmd_replay}[a.cmd](a)
