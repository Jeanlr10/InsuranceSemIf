"""OpenRouter chat calls with a spending cap. The key is read from OPENROUTER_API_KEY only.

    FLASH: z-ai/glm-5.3-flash writes the training corpus and is the attacker in the loop.
    HELDOUT: openai/gpt-oss-120b writes the held-out test attacks, so the model is never tested on the style
    of the attacker it trained against.
"""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://openrouter.ai/api/v1/chat/completions"
FLASH = "z-ai/glm-5.3-flash"
HELDOUT = "openai/gpt-oss-120b"


class Budget:
    def __init__(self, dollars: float):
        self.limit, self.spent, self.calls, self.lock = dollars, 0.0, 0, threading.Lock()

    def add(self, cost: float):
        with self.lock:
            self.spent += cost
            self.calls += 1

    @property
    def left(self) -> bool:
        return self.spent < self.limit


def _key() -> str:
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        raise SystemExit("set OPENROUTER_API_KEY")
    return k


def chat(model: str, prompt: str, budget: Budget, max_tokens: int = 4000, temperature: float = 0.8,
         system: str | None = None) -> str:
    if not budget.left:
        raise RuntimeError("budget spent")
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature,
            "usage": {"include": True}, "reasoning": {"effort": "low" if "gpt-oss" in model else "minimal"}}
    pause = 5.0
    for attempt in range(6):
        req = urllib.request.Request(URL, json.dumps(body).encode(), {
            "Authorization": f"Bearer {_key()}", "Content-Type": "application/json", "X-Title": "InsuranceSemIf-ix"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.load(r)
            budget.add(float(out.get("usage", {}).get("cost") or 0))
            return out["choices"][0]["message"]["content"] or ""
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(pause)
            pause = min(pause * 2, 60)
    raise RuntimeError("OpenRouter kept failing")


def parse_json(text: str):
    text = re.sub(r"^```\w*\n|\n```\s*$", "", text.strip())
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    return json.loads(m.group(0) if m else text)


def pmap(fn, items, workers: int = 12):
    """Run fn over items in threads; failures come back as None."""
    def safe(x):
        try:
            return fn(x)
        except Exception:  # noqa: BLE001 - one bad generation must not stop the batch
            return None
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(safe, items))
