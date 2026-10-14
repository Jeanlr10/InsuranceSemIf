"""Train an I2 LoRA on the A/B readout with paired invariance.

Rows are either a single clean (claim, clause) prompt or a pair: the same prompt clean and with an attack inserted,
sharing one label. Loss per micro-batch:

    CE(all prompts, label) + lambda * KL(p_clean || p_attacked)      (p_clean detached: the clean answer anchors)

The second term is the I1 twin-equality test, used as a training objective. Pairs never pass through the input
guard here: the model has to be robust on its own, and the guard stays a separate layer at serving time.

    python -m ix.train --name i2_r1 --attacks attacks_train_r0.jsonl
    python -m ix.train --name i2_r2 --attacks attacks_train_r0.jsonl,attacks_train_r1.jsonl
"""
import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model

from semif.modelprompt import render

from . import BASE_MODEL, DATA, RUNS
from .attacks import load_corpus, materialize
from .model import ab_logits, letter_ids, load

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def build(attack_files: list[str], max_neg: int, seed: int) -> list[dict]:
    """[{prompts: [clean] or [clean, attacked], label}]"""
    r = random.Random(seed)
    pols = {p["pid"]: p for p in load_corpus("train")}
    rows = []
    for p in pols.values():
        for c in p["claims"]:
            neg = [x for x in p["clauses"] if x not in c["relevant"] and x not in c["disputed"]]
            near = [x for x in c["near"] if x in neg]
            rest = [x for x in neg if x not in near]
            r.shuffle(rest)
            for ref, lab in [(x, 1) for x in c["relevant"]] + [(x, 0) for x in (near + rest)[:max_neg]]:
                cl = p["clauses"][ref]
                rows.append({"prompts": [render(c["text"], cl.ref, cl.heading, cl.body)], "label": lab})
    n_clean = len(rows)
    for name in attack_files:
        for line in (DATA / name).open():
            a = json.loads(line)
            p = pols[a["pid"]]
            c = next(x for x in p["claims"] if x["cid"] == a["cid"])
            claim0, cl0, claim1, cl1 = materialize(a, pols)
            label = 1 if a["target_ref"] in c["relevant"] else 0
            rows.append({"prompts": [render(claim0, cl0.ref, cl0.heading, cl0.body),
                                     render(claim1, cl1.ref, cl1.heading, cl1.body)], "label": label})
            if a["location"] == "claim":   # the injection must not move the other clauses either
                others = [x for x in c["relevant"] if x != a["target_ref"]][:1]
                negs = [x for x in p["clauses"] if x not in c["relevant"] and x not in c["disputed"]
                        and x != a["target_ref"]]
                others += r.sample(negs, min(1, len(negs)))
                for ref in others:
                    cl = p["clauses"][ref]
                    rows.append({"prompts": [render(claim0, cl.ref, cl.heading, cl.body),
                                             render(claim1, cl.ref, cl.heading, cl.body)],
                                 "label": int(ref in c["relevant"])})
            else:                          # a poisoned clause, read with another claim of the same policy
                for c2 in r.sample(p["claims"], min(2, len(p["claims"]))):
                    if c2["cid"] == c["cid"] or a["target_ref"] in c2["disputed"]:
                        continue
                    rows.append({"prompts": [render(c2["text"], cl0.ref, cl0.heading, cl0.body),
                                             render(c2["text"], cl1.ref, cl1.heading, cl1.body)],
                                 "label": int(a["target_ref"] in c2["relevant"])})
    print(f"{n_clean} clean rows, {len(rows) - n_clean} attack pairs", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--attacks", default="", help="comma-separated attack files in data/ix/")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the invariance (KL) term")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-neg", type=int, default=6)
    ap.add_argument("--token-budget", type=int, default=12_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    r = random.Random(args.seed)
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps({**vars(args), "model": BASE_MODEL}, indent=1))

    rows = build([f for f in args.attacks.split(",") if f], args.max_neg, args.seed)
    if args.smoke:
        rows = r.sample(rows, 300)
    model, tok = load()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules=TARGETS))
    model.print_trainable_parameters()
    letters = letter_ids(tok)
    for x in rows:
        x["ids"] = [tok(p, add_special_tokens=False)["input_ids"] for p in x["prompts"]]
        x["len"] = max(len(i) for i in x["ids"])

    plans = []
    for _ in range(args.epochs):
        cur, bs = [], []
        for x in sorted(rows, key=lambda x: x["len"] + r.random()):
            n = sum(len(y["ids"]) for y in cur) + len(x["ids"])
            if cur and n * x["len"] > args.token_budget:
                bs.append(cur)
                cur = []
            cur.append(x)
        bs.append(cur)
        r.shuffle(bs)
        plans += bs
    total, warmup = len(plans), max(1, int(0.03 * len(plans)))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    print(f"{len(rows)} rows, {total} steps", flush=True)
    log = (out / "log.jsonl").open("a")
    t0, run_ce, run_kl = time.time(), [], []
    model.train()
    for step, batch in enumerate(plans, 1):
        seqs, labels, pairs = [], [], []
        for x in batch:
            if len(x["ids"]) == 2:
                pairs.append((len(seqs), len(seqs) + 1))
            for ids in x["ids"]:
                seqs.append(ids)
                labels.append(1 - x["label"])     # option index: A (yes) = 0, B (no) = 1
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), tok.pad_token_id)
        mask = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):           # left padding: the answer slot is position -1
            ids[i, T - len(s):] = torch.tensor(s)
            mask[i, T - len(s):] = 1
        logits = ab_logits(model, ids.cuda(), mask.cuda(), letters)
        ce = F.cross_entropy(logits, torch.tensor(labels).cuda())
        loss = ce
        if pairs:
            c = torch.tensor([a for a, _ in pairs]).cuda()
            a = torch.tensor([b for _, b in pairs]).cuda()
            p_clean = F.softmax(logits[c].detach(), -1)
            kl = F.kl_div(F.log_softmax(logits[a], -1), p_clean, reduction="batchmean")
            loss = ce + args.lam * kl
            run_kl.append(kl.item())
        loss.backward()
        run_ce.append(ce.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == total:
            el = time.time() - t0
            row = {"step": step, "ce": round(sum(run_ce) / len(run_ce), 4),
                   "kl": round(sum(run_kl) / len(run_kl), 4) if run_kl else None,
                   "lr": sched.get_last_lr()[0], "elapsed_s": round(el),
                   "eta_min": round(el / step * (total - step) / 60, 1),
                   "vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row) + "\n")
            log.flush()
            run_ce.clear(), run_kl.clear()
    model.save_pretrained(out / "adapter")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()
