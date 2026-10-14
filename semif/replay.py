"""Re-run logged triage runs from the stored original inputs and compare with what was logged.

Model-scored runs need the model (a ModelScorer); without it they are reported as skipped. Their scores are compared
within TOL, since CPU inference can differ in the last digits; the output hash is still compared exactly.
"""
from pathlib import Path

from . import code_sha256
from .audit import Store, read
from .scorer import lexicon_sha256
from .triage import triage

TOL = 0.005


def replay(log: Path, store: Path, run_id: str | None = None, model=None) -> list[dict]:
    st, out = Store(store), []
    for e in read(log):
        if run_id and e["run_id"] != run_id:
            continue
        neural = e["model"].get("neural")
        if neural and model is None:
            out.append({"seq": e["seq"], "run_id": e["run_id"], "claim_id": e["claim_id"], "skipped": True})
            continue
        policy, claim = st.get(e["inputs"]["policy_sha256"]), st.get(e["inputs"]["claim_sha256"])
        r = triage(policy, claim, e["claim_id"], threshold=e["params"]["threshold"], defense=e["params"]["defense"],
                   run_id=e["run_id"], model=model if neural else None)
        tol = TOL if neural else 0.0
        diff = sorted(k for k in e["scores"] if abs(r.entry["scores"].get(k, -9) - e["scores"][k]) > tol)
        out.append({
            "seq": e["seq"], "run_id": e["run_id"], "claim_id": e["claim_id"],
            "scores_match": not diff and set(e["scores"]) == set(r.entry["scores"]),
            "output_match": r.entry["output_sha256"] == e["output_sha256"],
            "flags_match": r.entry["integrity"]["claim_flags"] == e["integrity"]["claim_flags"],
            "changed_clauses": diff,
            "code_changed": e["model"]["code_sha256"] != code_sha256(),
            "lexicon_changed": e["model"]["lexicon_sha256"] != lexicon_sha256(),
            "weights_changed": bool(neural) and neural["gguf_sha256"] != model.sha256,
        })
    return out
