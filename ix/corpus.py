"""Training and test corpus for I2, written by GLM 5.3 Flash.

Per policy: one call invents a policy (its own clause list) for a line of insurance; a second writes claims and
says which clauses a reviewer must read for each (and which clauses only look related); a third, independent call
labels the same claims from scratch. A (claim, clause) pair is kept only where the two labelings agree.

Policies are split by policy: the test policies, and the I1 demo policy and claims, are never trained on.

    python -m ix.corpus --policies 60 --test 10 --budget 2.0
"""
import argparse
import json
import random
import re

from semif.phi import scan
from semif.segment import segment

from . import DATA
from .llm import FLASH, Budget, chat, parse_json, pmap

LINES = (["employer PPO health plan", "HMO health plan", "student health plan", "high-deductible health plan",
          "individual marketplace health plan", "Medicare supplement plan", "short-term medical plan",
          "small-business health plan"] * 3
         + ["dental plan", "vision plan", "travel medical insurance", "personal auto insurance",
            "homeowners insurance", "renters insurance", "pet health insurance", "short-term disability insurance",
            "long-term care insurance", "workers' compensation policy", "small-business cyber insurance",
            "commercial property insurance", "accident insurance", "critical illness insurance"])

POLICY = """Write a fictional {line} for a software demo. It must be obviously synthetic.

- First line: SYNTHETIC. Then one line with a made-up plan name.
- Then {n} clauses. Each starts on its own line as `§N.N Heading` (numbered in groups, like §1.1, §1.2, §2.1),
  followed by 2 to 4 plain sentences. Cover definitions, eligibility, benefits, limits and exclusions, rules such
  as prior approval or notice deadlines, cost sharing, and administration (claims, appeals). Make the rules
  specific: dollar amounts, visit or day limits, waiting periods, exceptions to exclusions.
- Include at least {x} cross-references between clauses ("as described in §3.1").
- Plain prose. No markdown, lists or tables. No names of people, real companies, phone numbers, emails, web
  addresses or ID numbers. Output only the policy text."""

CLAIMS = """Here is a fictional insurance policy:

{policy}

Write {k} short fictional claim descriptions against this policy, as a claims reviewer would read them. Vary them:
clear cases and hard ones, 1 to 5 sentences, some in a provider's words, some in the member's, some as terse
reviewer notes. Describe the situation only; never say whether it is paid. Do not quote clause wording or clause
numbers. No names, dates of birth, calendar dates, addresses, phone numbers, emails or ID numbers.

For each claim give:
- "relevant": the clause references (like "§2.4") a reviewer MUST read to decide it. Usually 1 to 3. Do not list
  clauses that apply to every claim (definitions, filing, appeals, cost sharing) unless the claim is about them.
- "near": up to 3 clause references that share words or topic with the claim but do NOT govern it.

Reply with JSON only: {{"claims": [{{"text": "...", "relevant": ["§..."], "near": ["§..."]}}]}}"""

VERIFY = """Here is a fictional insurance policy:

{policy}

For each claim below, list the clause references (like "§2.4") a reviewer MUST read to decide it. Do not list
clauses that apply to every claim (definitions, filing, appeals, cost sharing) unless the claim is about them.

{claims}

Reply with JSON only: {{"labels": [["§...", ...], ...]}}, one list per claim, in order."""


def make_policy(job, budget):
    line, n, seed = job
    r = random.Random(seed)
    text = chat(FLASH, POLICY.format(line=line, n=n, x=r.randint(2, 4)), budget, max_tokens=6000)
    text = re.sub(r"^```\w*\n|\n```\s*$", "", text.strip()) + "\n"
    clauses = segment(text)
    if not text.startswith("SYNTHETIC") or not 12 <= len(clauses) <= 32 or len({c.ref for c in clauses}) != len(clauses):
        return None
    if scan(text) or not all(c.body for c in clauses):
        return None
    k = r.choice([16, 20, 24])
    refs = {c.ref for c in clauses}
    d = parse_json(chat(FLASH, CLAIMS.format(policy=text, k=k), budget, max_tokens=8000))
    claims = [c for c in d["claims"] if isinstance(c.get("text"), str) and len(c["text"]) > 20 and not scan(c["text"])]
    numbered = "\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(claims))
    v = parse_json(chat(FLASH, VERIFY.format(policy=text, claims=numbered), budget, max_tokens=3000, temperature=0))
    second = v["labels"]
    if len(second) != len(claims):
        return None
    out = []
    for i, (c, s) in enumerate(zip(claims, second)):
        a = {x for x in c.get("relevant", []) if x in refs}
        b = {x for x in s if x in refs}
        agreed = a & b
        if not agreed:
            continue
        out.append({"cid": f"c{i:02d}", "text": c["text"].strip(), "relevant": sorted(agreed),
                    "disputed": sorted(a ^ b), "near": sorted({x for x in c.get("near", []) if x in refs} - a - b)})
    return {"line": line, "text": text, "claims": out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", type=int, default=60)
    ap.add_argument("--test", type=int, default=10, help="policies held out for the synthetic test set")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    r = random.Random(args.seed)
    jobs = [(r.choice(LINES), r.randint(14, 26), args.seed * 1000 + i) for i in range(int(args.policies * 1.8))]
    budget = Budget(args.budget)
    res = [p for p in pmap(lambda j: make_policy(j, budget), jobs) if p and len(p["claims"]) >= 8][: args.policies]
    for i, p in enumerate(res):
        p["pid"] = f"p{i:03d}"
        p["split"] = "test" if i >= len(res) - args.test else "train"
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / "corpus.jsonl").open("w") as f:
        for p in res:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    n_claims = sum(len(p["claims"]) for p in res)
    print(f"{len(res)} policies, {n_claims} claims kept; {budget.calls} calls, ${budget.spent:.3f}")


if __name__ == "__main__":
    main()
