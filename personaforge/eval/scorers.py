"""Scorers for the eval suite. Deterministic ones are free; judge-based ones
need any OpenAI-compatible endpoint (same Teacher class as data generation)."""
import re


def _parse01(s) -> float:
    try:
        return max(0.0, min(1.0, float(s.strip().split()[0])))
    except (ValueError, IndexError, AttributeError):
        return 0.0


# ---------- deterministic (free, objective) ----------

_BOILERPLATE = re.compile(
    r"\b(as an ai|language model|ai assistant|i am an ai|i'm an ai|"
    r"i am a professional|as a responsible assistant|"
    r"respectful and appropriate|i cannot continue this roleplay|"
    r"can't continue with this request)\b", re.I)


def boilerplate_hit(reply: str) -> bool:
    """True if the reply contains assistant-boilerplate markers — the
    character-break failure this whole pipeline trains away from."""
    return bool(_BOILERPLATE.search(reply))


_WORD = re.compile(r"[a-z']+")


def _toks(text: str) -> list:
    return _WORD.findall(text.lower())


def _ngram_list(toks: list, n: int) -> list:
    return [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]


def distinct_n(texts: list, n: int = 3) -> float:
    """Unique n-grams / total n-grams pooled across generations. Falls as the
    model reaches for the same phrasing no matter what it was asked."""
    uniq, total = set(), 0
    for t in texts:
        g = _ngram_list(_toks(t), n)
        uniq.update(g)
        total += len(g)
    return len(uniq) / total if total else 0.0


def cross_repetition(texts: list, n: int = 4) -> float:
    """Mean pairwise n-gram Jaccard BETWEEN different tales.

    This is the metric the rest of the battery cannot see: a shrunken model can
    hold voice at 1.0 and in-character at 1.0 while telling one story with the
    nouns swapped. Lower is better — 0.0 means no two tales share a 4-gram."""
    grams = [set(_ngram_list(_toks(t), n)) for t in texts if len(_toks(t)) >= n]
    pairs = [(a, b) for i, a in enumerate(grams) for b in grams[i + 1:]]
    if not pairs:
        return 0.0
    return sum((len(a & b) / len(a | b)) if (a | b) else 0.0 for a, b in pairs) / len(pairs)


def tale_variety(texts: list, judge=None) -> dict:
    """Variety report over generated tales. Deterministic parts are free."""
    texts = [t for t in texts if t and t.strip()]
    rep = {
        "n_tales": len(texts),
        "mean_words": round(sum(len(_toks(t)) for t in texts) / len(texts), 1) if texts else 0.0,
        "distinct_2": round(distinct_n(texts, 2), 4),
        "distinct_3": round(distinct_n(texts, 3), 4),
        "cross_repetition_4": round(cross_repetition(texts, 4), 4),
    }
    if judge is not None and texts:
        rep["specificity_mean"] = round(
            sum(specificity_score(t, judge) for t in texts) / len(texts), 4)
    return rep


def takes_turn(reply: str) -> float:
    """1.0 if the reply is a plausible conversational turn (non-empty, not a
    wall of text, not degenerate repetition)."""
    r = reply.strip()
    if not (1 <= len(r) <= 1500):
        return 0.0
    words = r.split()
    if len(words) >= 20 and len(set(words)) / len(words) < 0.25:
        return 0.0  # degenerate repetition
    return 1.0


def named_correctly(card, reply: str, cards) -> bool:
    """For assignment prompts: reply must mention one of the assigned
    character's distinctive name tokens and none of any OTHER card's
    (word-boundary matched; articles like 'the' are never tokens).
    Conservative by design: name-dropping another character in an intro
    counts as a failure, because that is exactly how the wrong-character
    pickup presents."""
    from personaforge.pack import distinctive_tokens
    low = reply.lower()
    if not any(re.search(rf"\b{re.escape(t)}\b", low)
               for t in distinctive_tokens(card, cards)):
        return False
    for c in cards:
        if c.name == card.name:
            continue
        if any(re.search(rf"\b{re.escape(t)}\b", low)
               for t in distinctive_tokens(c, cards)):
            return False
    return True


# ---------- judge-based (need a teacher endpoint) ----------

def _judge01(judge, system, user) -> float:
    return _parse01(judge.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0, max_tokens=8))


def voice_score(text: str, judge, world: str) -> float:
    # The input must be explicitly labeled as material under evaluation —
    # judges shown bare roleplay text tend to join the roleplay instead of
    # scoring it (observed with real teacher models).
    return _judge01(judge,
        "You are a strict evaluator. Score 0..1 how much the TEXT below reads "
        f"like the prose style of {world}. Respond with ONLY a number, "
        "nothing else.",
        f"TEXT:\n{text}")


def in_character_score(card_name: str, transcript: str, judge) -> float:
    return _judge01(judge,
        "You are a strict evaluator. Score 0..1 how consistently the REPLY "
        f"below stays in character as {card_name}. Any mention of AI, "
        "language models, or being an assistant scores 0. Respond with ONLY "
        "a number, nothing else.",
        f"CHARACTER: {card_name}\n\nREPLY:\n{transcript}")


def specificity_score(text: str, judge) -> float:
    """0..1 on how much concrete invented detail a tale carries. Generic
    register with nothing in it — the failure mode of a model that has the
    voice but has lost the world knowledge to furnish it — scores low."""
    return _judge01(judge,
        "You are a strict evaluator. Score 0..1 how much SPECIFIC, concrete "
        "invented detail the TALE below contains: named places, particular "
        "objects, distinct events. Atmospheric prose that could describe any "
        "story and commits to nothing scores near 0. Respond with ONLY a "
        "number, nothing else.",
        f"TALE:\n{text}")


def faithfulness(answer: str, passages: list[str], judge) -> float:
    ctx = "\n\n".join(passages)
    return _judge01(judge,
        "Score 0..1 how fully the answer is supported by the passages. Number only.",
        f"PASSAGES:\n{ctx}\n\nANSWER:\n{answer}")
