"""Decontamination gate for teacher outputs derived from private material.

Two mechanical checks, both conservative:
- n-gram overlap: any run of `n` consecutive words shared with the reference
  corpus rejects the sample (catches memorized expression surfacing through a
  tuned teacher).
- blocklist: any word-boundary hit on a protected coinage rejects the sample.

The corpus and blocklist are runtime inputs — this module ships no protected
content. `report()` feeds the provenance report that accompanies a release.
"""
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Decontaminator:
    def __init__(self, corpus_text: str, blocklist: list[str], n: int = 8):
        self.n = n
        words = _words(corpus_text)
        self._corpus_ngrams = {tuple(words[i:i + n])
                               for i in range(max(0, len(words) - n + 1))}
        self._block = [re.compile(rf"\b{re.escape(b.lower())}\b")
                       for b in blocklist]
        self._block_names = [b.lower() for b in blocklist]
        self._checked = 0
        self._dropped: Counter = Counter()

    def check(self, text: str) -> tuple[bool, str]:
        self._checked += 1
        low = text.lower()
        for pat, name in zip(self._block, self._block_names):
            if pat.search(low):
                reason = f"blocklist:{name}"
                self._dropped[reason] += 1
                return False, reason
        words = _words(text)
        for i in range(max(0, len(words) - self.n + 1)):
            if tuple(words[i:i + self.n]) in self._corpus_ngrams:
                self._dropped["ngram_overlap"] += 1
                return False, "ngram_overlap"
        return True, "ok"

    def report(self) -> dict:
        return {"checked": self._checked,
                "ok": self._checked - sum(self._dropped.values()),
                "n": self.n,
                "dropped": dict(self._dropped)}
