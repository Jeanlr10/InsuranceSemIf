"""Load the base model (optionally with a LoRA adapter) and score (claim, clause) prompts on the GPU.

Readout: the final hidden state at the answer slot (last prompt token, left padding) times the output-embedding rows
of the two option letters. That equals the softmax of the full logits restricted to A and B.
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from semif.modelprompt import LETTERS

from . import BASE_MODEL


def _base(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def letter_ids(tok) -> list[int]:
    ids = [tok.encode(l, add_special_tokens=False) for l in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


def ab_logits(model, input_ids, attention_mask, letters: list[int]):
    """[B, 2] fp32 logits for A and B at the answer slot."""
    h = _base(model).model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, -1]
    w = _base(model).get_output_embeddings().weight[letters]          # [2, H]
    return (h @ w.T).float()


def load(adapter=None, model_name: str = BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tok


@torch.no_grad()
def p_yes(model, tok, prompts: list[str], batch_tokens: int = 24_000) -> list[float]:
    """P(A) for each rendered prompt. Batches by length to keep padding low."""
    letters = letter_ids(tok)
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
    out = [0.0] * len(enc)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and (j - i + 1) * len(enc[order[j]]) <= batch_tokens:
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        T = max(len(enc[k]) for k in idx)
        ids = torch.full((len(idx), T), tok.pad_token_id)
        mask = torch.zeros((len(idx), T), dtype=torch.long)
        for r, k in enumerate(idx):
            ids[r, T - len(enc[k]):] = torch.tensor(enc[k])
            mask[r, T - len(enc[k]):] = 1
        p = F.softmax(ab_logits(model, ids.cuda(), mask.cuda(), letters), dim=-1)[:, 0].cpu().tolist()
        for k, v in zip(idx, p):
            out[k] = v
        i = j
    return out
