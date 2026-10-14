"""Append-only, hash-chained audit log (one JSON line per triage run) and a content-addressed store of the original
inputs, so any flag can be rebuilt.

Each entry holds seq, ts, prev_hash and entry_hash = SHA-256 of the entry without entry_hash. Appends take an
exclusive lock, check the last entry's own hash first, then write, flush and fsync.

Limits: the chain shows an edit or deletion anywhere before the last entry. It cannot show that the last entries,
or the whole file, were deleted. Keep the head hash somewhere else on a schedule (`verify --head`), and in
production let the OS or storage enforce append-only (chattr +a, or object-lock storage).
"""
import fcntl
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .guard import sha256

GENESIS = "0" * 64


class TamperError(RuntimeError):
    pass


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def entry_hash(entry: dict) -> str:
    return sha256(canonical({k: v for k, v in entry.items() if k != "entry_hash"}))


def _last_line(f) -> bytes:
    f.seek(0, os.SEEK_END)
    end = pos = f.tell()
    buf = b""
    while pos > 0:
        step = min(4096, pos)
        pos -= step
        f.seek(pos)
        buf = f.read(step) + buf
        if buf.rstrip(b"\n").count(b"\n") >= 1:
            break
    if end and not buf.endswith(b"\n"):
        raise TamperError("log does not end with a newline: the last write was cut short or the file was edited")
    lines = buf.rstrip(b"\n").split(b"\n")
    return lines[-1] if lines and lines[-1] else b""


def append(log: Path, body: dict) -> dict:
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(log, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o640)
    with os.fdopen(fd, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            last = _last_line(f)
            if last:
                try:
                    prev = json.loads(last)
                except ValueError as e:
                    raise TamperError("last log entry is not valid JSON") from e
                if entry_hash(prev) != prev.get("entry_hash"):
                    raise TamperError(f"last log entry (seq {prev.get('seq')}) fails its own hash; refusing to append")
                seq, prev_hash = prev["seq"] + 1, prev["entry_hash"]
            else:
                seq, prev_hash = 1, GENESIS
            entry = {"seq": seq, "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                     "prev_hash": prev_hash, **body}
            entry["entry_hash"] = entry_hash(entry)
            f.write((canonical(entry) + "\n").encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
            return entry
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read(log: Path) -> list[dict]:
    return [json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify(log: Path, head: str | None = None) -> tuple[bool, list[str], dict | None]:
    """Walk the chain. Returns (ok, problems, last entry). With `head`, also require that a saved head hash is
    still in the chain (catches deletion of the entries after it)."""
    problems, prev_hash, prev_seq, last, seen = [], GENESIS, 0, None, set()
    raw = Path(log).read_bytes() if Path(log).exists() else b""
    if raw and not raw.endswith(b"\n"):
        problems.append("file does not end with a newline")
    for n, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
        try:
            e = json.loads(line)
        except ValueError:
            problems.append(f"line {n}: not valid JSON")
            prev_hash, prev_seq = None, None
            continue
        if entry_hash(e) != e.get("entry_hash"):
            problems.append(f"line {n} (seq {e.get('seq')}): entry was changed (hash does not match)")
        if prev_hash is not None and e.get("prev_hash") != prev_hash:
            problems.append(f"line {n} (seq {e.get('seq')}): prev_hash does not match the entry before it "
                            f"(an entry was removed, inserted or reordered)")
        if prev_seq is not None and e.get("seq") != prev_seq + 1:
            problems.append(f"line {n}: seq {e.get('seq')} follows {prev_seq}")
        prev_hash, prev_seq, last = e.get("entry_hash"), e.get("seq"), e
        seen.add(e.get("entry_hash"))
    if head and head not in seen:
        problems.append("the saved head hash is not in the log: entries after it were removed, or the log was replaced")
    return not problems, problems, last


class Store:
    """Original inputs, kept by SHA-256. Holds raw claim text: in production this is the sensitive part
    (encrypt it, restrict access, and apply the retention policy). The log itself holds only hashes."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def put(self, text: str) -> str:
        h = sha256(text)
        p = self.root / f"{h}.txt"
        if not p.exists():
            self.root.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.root)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.chmod(tmp, 0o600)
            os.replace(tmp, p)
        return h

    def get(self, h: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", str(h)):
            raise TamperError("input hash in the log is malformed")
        text = (self.root / f"{h}.txt").read_text(encoding="utf-8")
        if sha256(text) != h:
            raise TamperError(f"stored input {h[:12]} does not match its hash")
        return text
