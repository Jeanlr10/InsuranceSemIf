"""Tokenizer and a small suffix-stripping stemmer. Letters only; numbers carry no relevance signal here."""
import re

STOP = set("""
a about above after again against all also am an and any are as at be because been before being below between both
but by can could did do does doing done down during each either else even ever every few for from further had has
have having he her here hers him his how however i if in into is it its itself just less may might more most much
must my no nor not now of off on once one only or other our out over own per same she should so some such than that
the their them then there these they this those through to too under until up upon us very via was we were what when
where whether which while who whom whose why will with within without would yet you your
claim claims member members plan plans patient patients policy clause clauses section sample demo record records
pay pays paid payment payments payable bill billed billing cost costs
covered cover covers coverage denied deny denial approved approve excluded exclude
""".split())
# The last line holds decision words. They are dropped on both sides, so they can never move a score, and so the
# rationale can never print them.

DECISION_WORDS = ("covered", "denied", "approved", "excluded")


def stem(w: str) -> str:
    if len(w) <= 3:
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    for suf in ("ing", "ed"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[:-len(suf)]
            break
    if w.endswith("e") and len(w) > 3:  # service/services, image/imaging
        w = w[:-1]
    return w


def words(text: str) -> list[str]:
    """Lower-case surface words, with hyphenated words joined (x-ray -> xray, off-label -> offlabel)."""
    t = re.sub(r"(?<=[a-z])-(?=[a-z])", "", text.lower())
    return [w for w in re.findall(r"[a-z]+", t) if len(w) >= 2]


def terms(text: str, keep_stop: bool = False) -> list[tuple[str, str]]:
    """(surface, stem) pairs, stop words removed."""
    return [(w, stem(w)) for w in words(text) if keep_stop or w not in STOP]
