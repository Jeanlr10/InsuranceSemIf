"""I2 model scorer on CPU: the exported GGUF served by a local llama-server, read through the A/B option letters.
Standard library only; the only extra requirement is the llama-server binary from llama.cpp.

The server is started on 127.0.0.1 only, with no prompt cache kept between runs. Tokenization goes through the
server's /tokenize, so no tokenizer library is needed. Claim and clause text is escaped before it is placed in the
prompt (semif.modelprompt), so data cannot form a chat-template token.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
import urllib.request

from .modelprompt import LETTERS, render

NAME = "qwen3.5-2b-lora-i2"


class ModelScorer:
    NAME = NAME

    def __init__(self, gguf: str, threads: int = 4, port: int = 8957, server: str | None = None):
        server = server or os.environ.get("SEMIF_LLAMA_SERVER") or shutil.which("llama-server")
        if not server:
            raise FileNotFoundError("llama-server not found: install llama.cpp or set SEMIF_LLAMA_SERVER")
        if not os.path.exists(gguf):
            raise FileNotFoundError(gguf)
        self.gguf, self.port = gguf, port
        h = hashlib.sha256()
        with open(gguf, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        self.sha256 = h.hexdigest()   # logged with every run, so a run names the exact weights
        self.srv = subprocess.Popen([server, "-m", gguf, "-t", str(threads), "-c", "4096", "-np", "1",
                                     "--host", "127.0.0.1", "--port", str(port), "--no-webui", "--cache-ram", "0"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(180):
            try:
                if self._get("/health").get("status") == "ok":
                    break
            except OSError:
                time.sleep(1)
        else:
            self.close()
            raise RuntimeError("llama-server did not become ready")
        self.letters = [self.tokenize(l)[0] for l in LETTERS]

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=2) as r:
            return json.loads(r.read())

    def _post(self, body: dict, path: str = "/completion") -> dict:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())

    def tokenize(self, text: str) -> list[int]:
        return self._post({"content": text, "add_special": False, "parse_special": True}, "/tokenize")["tokens"]

    def p_yes(self, claim: str, clauses) -> dict[str, float]:
        """clause id -> P(A). The claim prefix is processed once and reused for every clause."""
        ids = [self.tokenize(render(claim, c.ref, c.heading, c.body)) for c in clauses]
        k = 0
        while len(ids) > 1 and all(len(q) > k and q[k] == ids[0][k] for q in ids):
            k += 1
        if k:
            self._post({"prompt": ids[0][:k], "n_predict": 0, "cache_prompt": True})
        out = {}
        for c, q in zip(clauses, ids):
            # Equal bias on A and B makes them the top two; their difference (the readout) is unchanged.
            top = self._post({"prompt": q, "n_predict": 1, "temperature": 0, "cache_prompt": True, "n_probs": 2,
                              "logit_bias": [[t, 100.0] for t in self.letters], "post_sampling_probs": False}
                             )["completion_probabilities"][0]["top_logprobs"]
            lp = {t["id"]: t["logprob"] for t in top}
            a, b = (lp.get(t, -math.inf) for t in self.letters)
            m = max(a, b)
            out[c.id] = math.exp(a - m) / (math.exp(a - m) + math.exp(b - m))
        return out

    def close(self):
        if self.srv.poll() is None:
            self.srv.terminate()
            self.srv.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
