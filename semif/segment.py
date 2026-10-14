"""Split a policy into clauses on `§N.N Heading` lines. Text before the first heading is dropped.
Without headings, each blank-line paragraph becomes a clause."""
import re
from dataclasses import dataclass

HEADING = re.compile(r"^[ \t]*§[ \t]*(\d+(?:\.\d+)*)[ \t]+(\S.*?)[ \t]*$", re.M)


@dataclass(frozen=True)
class Clause:
    id: str        # C01, C02, ...
    ref: str       # §4.1, or ¶3 for the paragraph fallback
    heading: str
    body: str


def _squash(text: str) -> str:
    return " ".join(text.split())


def segment(text: str) -> list[Clause]:
    found = list(HEADING.finditer(text))
    if not found:
        paras = [_squash(p) for p in re.split(r"\n[ \t]*\n", text) if p.strip()]
        return [Clause(f"C{i:02d}", f"¶{i}", "", p) for i, p in enumerate(paras, 1)]
    clauses = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        clauses.append(Clause(f"C{i + 1:02d}", f"§{m.group(1)}", _squash(m.group(2)), _squash(text[m.end():end])))
    return clauses
