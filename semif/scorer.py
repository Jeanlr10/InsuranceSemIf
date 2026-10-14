"""Deterministic lexical scorer: BM25 over clauses, heading words counted twice, each claim word counted once, and a
small lexicon of related words at a lower weight. It has no instruction-following path: the only way text can move
a score is by sharing words with a clause."""
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass

from . import PACKAGE
from .text import terms

NAME = "lexical-bm25"
VERSION = "1.0"
# Fixed before the eval was run. Not tuned afterwards.
PARAMS = {"k1": 1.2, "b": 0.75, "heading_weight": 2, "lexicon_weight": 0.6, "squash_k": 6.0}
LEXICON_PATH = PACKAGE / "lexicon.json"


def lexicon_sha256() -> str:
    return hashlib.sha256(LEXICON_PATH.read_bytes()).hexdigest()


def load_lexicon() -> dict[str, list[tuple[str, str]]]:
    """claim stem -> [(policy stem, lexicon key as written)]."""
    entries = json.loads(LEXICON_PATH.read_text())["entries"]
    out: dict[str, list] = {}
    for key, targets in entries.items():
        for _, k in terms(key, keep_stop=True):
            for t in targets:
                out.setdefault(k, []).extend((s, key) for _, s in terms(t, keep_stop=True))
    return out


@dataclass
class Term:
    stem: str
    surface: str      # the word as it appears in the clause
    weight: float     # 1.0 for a shared word, lexicon_weight for a related word
    via: str | None   # the lexicon key that matched, or None
    in_heading: bool
    contribution: float


@dataclass
class ClauseScore:
    score: float
    raw: float
    terms: list[Term]


class Scorer:
    def __init__(self, clauses, skip=frozenset(), lexicon=None):
        """clauses: Clause objects. skip: clause IDs that failed the integrity check; they are not indexed."""
        self.clauses = [c for c in clauses if c.id not in skip]
        self.skip = set(skip)
        self.lexicon = load_lexicon() if lexicon is None else lexicon
        p = PARAMS
        self.tf, self.surface, self.heading = {}, {}, {}
        for c in self.clauses:
            h, b = terms(c.heading), terms(c.body)
            tf = Counter(s for _, s in b)
            for _, s in h:
                tf[s] += p["heading_weight"]
            self.tf[c.id] = tf
            self.heading[c.id] = {s for _, s in h}
            surf = {}
            for w, s in h + b:
                surf.setdefault(s, w)
            self.surface[c.id] = surf
        self.n = len(self.clauses)
        self.avgdl = sum(sum(t.values()) for t in self.tf.values()) / max(self.n, 1)
        self.df = Counter(s for t in self.tf.values() for s in t)

    def idf(self, s: str) -> float:
        df = self.df.get(s, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def query(self, claim_text: str) -> dict[str, tuple[float, str | None]]:
        """stem -> (weight, via). Each claim word counts once, however often it repeats."""
        q = {s: (1.0, None) for _, s in terms(claim_text)}
        for s in list(q):
            for target, key in self.lexicon.get(s, []):
                if target not in q:
                    q[target] = (PARAMS["lexicon_weight"], key)
        return q

    def score(self, claim_text: str) -> dict[str, ClauseScore]:
        p, q = PARAMS, self.query(claim_text)
        out = {}
        for c in self.clauses:
            tf, dl = self.tf[c.id], sum(self.tf[c.id].values())
            norm = p["k1"] * (1 - p["b"] + p["b"] * dl / self.avgdl)
            found, raw = [], 0.0
            for s, (w, via) in q.items():
                f = tf.get(s, 0)
                if not f:
                    continue
                part = w * self.idf(s) * f * (p["k1"] + 1) / (f + norm)
                raw += part
                found.append(Term(s, self.surface[c.id][s], w, via, s in self.heading[c.id], part))
            found.sort(key=lambda t: (-t.contribution, t.stem))
            out[c.id] = ClauseScore(1 - math.exp(-raw / p["squash_k"]), raw, found)
        return out
