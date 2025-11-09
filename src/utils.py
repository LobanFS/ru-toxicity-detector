import re
from typing import Iterable

RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_USER = re.compile(r"@\w+")
RE_HASH = re.compile(r"#\w+")
RE_SPACES = re.compile(r"\s+")
RE_STRETCH = re.compile(r"(.)\1{3,}")

def clean_for_tfidf(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = text.replace("Ё", "Е").replace("ё", "е")
    t = RE_URL.sub(" ", t)
    t = RE_USER.sub(" ", t)
    t = RE_HASH.sub(" ", t)
    t = RE_STRETCH.sub(r"\1\1", t)
    t = RE_SPACES.sub(" ", t).strip()
    return t.lower()

def iter_clean(xs: Iterable[str]):
    for x in xs:
        yield clean_for_tfidf(x)