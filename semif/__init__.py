"""InsuranceSemIf: rank the clauses of a policy by likely relevance to a claim. Triage only; a person decides."""
import hashlib
from pathlib import Path

VERSION = "1.0.0"
PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent


def code_sha256() -> str:
    """Hash of every source file in the package, so a log entry names the exact code that made it."""
    h = hashlib.sha256()
    for f in sorted(PACKAGE.glob("*.py")):
        h.update(f.name.encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()
