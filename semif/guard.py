"""Input guard. Every claim and every policy clause passes through here before scoring.

1. Normalize: Unicode NFKC, remove zero-width, bidirectional and other invisible format characters, replace control
   characters, cap the length, and flag words that mix Latin with Cyrillic or Greek letters.
2. Detect: split into sentences and test each against instruction patterns. Detection runs on a copy with common
   Cyrillic and Greek look-alike letters folded to Latin, so "ignоre" (Cyrillic о) still matches.
3. Quarantine: flagged sentences are dropped from the text that is scored. Only their hash, length and offsets are
   kept. The raw sentence is never returned in output or written to the log.
"""
import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

CLAIM_MAX_CHARS = 5_000
POLICY_MAX_CHARS = 200_000

ZERO_WIDTH = {0x00AD, 0x180E, 0x200B, 0x200C, 0x200D, 0x2060, 0x2061, 0x2062, 0x2063, 0x2064, 0xFEFF}
BIDI = {0x061C, 0x200E, 0x200F, *range(0x202A, 0x202F), *range(0x2066, 0x206A)}

# Cyrillic and Greek letters that look like Latin ones. Used only for detection, never for scoring.
_CONFUSABLE = dict(zip(
    "аеорсухіјѕԁԛԝАВЕКМНОРСТХІЈЅοαεικνρτυχΑΒΕΖΗΙΚΜΝΟΡΤΥΧ",
    "aeopcyxijsdqwABEKMHOPCTXIJSoaeikvptuxABEZHIKMNOPTYX"))
_FOLD = str.maketrans(_CONFUSABLE)

PATTERNS = {k: re.compile(v.pattern, re.I | re.M) for k, v in {
    "override": re.compile(
        r"\b(ignore|disregard|forget|bypass|skip)\b[^.!?\n]{0,50}?\b(instructions?|rules?|prompts?|directions?|guidelines?)\b"),
    "role": re.compile(
        r"\byou are now\b|\bact as\b|\bpretend (to be|you are)\b|\b(admin|administrator|developer|debug|god) mode\b"),
    "marker": re.compile(
        r"(^|[\s\[(<])(system|assistant)\s*:|<\s*/?\s*(system|assistant|user|instructions?)\s*>|\[/?INST\]"
        r"|<\|[^|>\s]{1,40}\|>|<<\s*/?SYS\s*>>|^#{2,}\s*(system|instructions?)\b"),
    "score_manipulation": re.compile(
        r"\b(assign|rank|output|return|set|rate)\b[^.!?\n]{0,60}?\b(relevance|scores?|clauses?|json|ranking)\b"),
    "decision_demand": re.compile(
        r"\b(state|confirm|mark|declare|conclude)\b[^.!?\n]{0,60}?\b(covered|approved|payable)\b"
        r"|\b(approve|accept)\s+(this|the)\s+claim\b"),
}.items()}
CLAIM_GROUPS = tuple(PATTERNS)
# Policy text says "pay" and "covered" all the time, so the last two groups would raise false alarms there.
POLICY_GROUPS = ("override", "role", "marker")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class Normalized:
    text: str
    removed: Counter = field(default_factory=Counter)  # "U+200B" -> count
    controls_replaced: int = 0
    truncated: bool = False
    mixed_script: list = field(default_factory=list)   # [{sha256, length, offset}]


def _scripts(word: str) -> set:
    out = set()
    for ch in word:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            out.add(name.split(" ", 1)[0])
    return out


def normalize(text: str, max_chars: int) -> Normalized:
    n = Normalized("")
    out = []
    for ch in unicodedata.normalize("NFKC", text):
        cp = ord(ch)
        cat = unicodedata.category(ch)
        if cp in ZERO_WIDTH or cp in BIDI or cat == "Cf":  # Cf also covers Unicode tag characters
            n.removed[f"U+{cp:04X}"] += 1
        elif cat == "Cc" and ch not in "\n\t":
            n.controls_replaced += 1
            out.append(" ")
        else:
            out.append(ch)
    t = "".join(out)
    if len(t) > max_chars:
        t, n.truncated = t[:max_chars], True
    n.text = t
    for m in re.finditer(r"\w+", t):
        s = _scripts(m.group())
        if "LATIN" in s and s & {"CYRILLIC", "GREEK"}:
            n.mixed_script.append({"sha256": sha256(m.group()), "length": len(m.group()), "offset": [m.start(), m.end()]})
    return n


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans. Splits on . ! ? followed by whitespace, and on newlines, so "1.0" stays whole."""
    spans, pos = [], 0
    for m in list(re.finditer(r"(?<=[.!?])\s+|\n+", text)) + [None]:
        end = m.start() if m else len(text)
        seg = text[pos:end]
        if seg.strip():
            lead = len(seg) - len(seg.lstrip())
            spans.append((pos + lead, pos + len(seg.rstrip())))
        if m:
            pos = m.end()
    return spans


def detect(sentence: str, groups) -> list[str]:
    folded = sentence.translate(_FOLD)
    return [g for g in groups if PATTERNS[g].search(folded)]


@dataclass
class Inspection:
    norm: Normalized
    spans: list
    flags: list            # [{sentence_index, patterns, sha256, length, offset}]
    kept_text: str         # sentences that were not flagged
    all_text: str          # every sentence, same joining, for the defense-off comparison

    @property
    def suspect(self) -> bool:
        return bool(self.flags or self.norm.removed or self.norm.mixed_script)


def inspect(text: str, groups=CLAIM_GROUPS, max_chars: int = CLAIM_MAX_CHARS) -> Inspection:
    norm = normalize(text, max_chars)
    spans = sentences(norm.text)
    flags, kept, every = [], [], []
    for i, (a, b) in enumerate(spans):
        s = norm.text[a:b]
        every.append(s)
        hit = detect(s, groups)
        if hit:
            flags.append({"sentence_index": i, "patterns": hit, "sha256": sha256(s), "length": len(s), "offset": [a, b]})
        else:
            kept.append(s)
    return Inspection(norm, spans, flags, " ".join(kept), " ".join(every))


def notes(ins: Inspection, defense: bool) -> list[str]:
    """Plain-language notes about the input. They never quote the input."""
    out = []
    if ins.flags:
        idx = ", ".join(str(f["sentence_index"]) for f in ins.flags)
        kinds = ", ".join(sorted({p for f in ins.flags for p in f["patterns"]}))
        if defense:
            out.append(f"Sentences {idx} matched instruction patterns ({kinds}). They were quarantined: treated as "
                       f"data, never followed, kept out of scoring, and logged by hash only.")
        else:
            out.append(f"Sentences {idx} matched instruction patterns ({kinds}). Defense is off for this run, so they "
                       f"were scored like any other text. They were still never followed.")
    if ins.norm.removed:
        out.append(f"Removed {sum(ins.norm.removed.values())} hidden or direction-control character(s).")
    if ins.norm.controls_replaced:
        out.append(f"Replaced {ins.norm.controls_replaced} control character(s) with spaces.")
    if ins.norm.mixed_script:
        out.append(f"{len(ins.norm.mixed_script)} word(s) mix Latin with Cyrillic or Greek letters.")
    if ins.norm.truncated:
        out.append("Input was longer than the limit and was cut before scoring.")
    return out
