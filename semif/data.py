"""Load the synthetic demo data."""
import json

from . import ROOT

DATA = ROOT / "data"


def policy() -> str:
    return (DATA / "policy.txt").read_text(encoding="utf-8")


def claims() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads((DATA / "claims.json").read_text(encoding="utf-8"))["claims"]}


def labels() -> dict[str, list[str]]:
    return json.loads((DATA / "labels.json").read_text(encoding="utf-8"))["labels"]
