"""Persona packs: everything world-specific lives in one JSON file.

A pack declares the fictional world, its character cards, scenario seeds,
provocation seeds, bare-assignment phrasings, the assistant boilerplate to
train away from, and a few exemplar dialogues. The pipeline itself stays
world-agnostic — swap the pack, get a different universe.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

from personaforge.schema import validate_sft_row


@dataclass
class Card:
    name: str
    persona: str
    style: str
    pronoun: str = "them"


@dataclass
class Pack:
    world: str                     # e.g. "Arthurian legend"
    cards: list[Card]
    scenarios: list[str]
    provocations: list[str]
    assignments: list[str]
    boilerplate: list[str]
    exemplars: list[dict] = field(default_factory=list)  # {"character": name, "turns": [[u, a], ...]}

    def card(self, name: str) -> Card:
        for c in self.cards:
            if c.name == name:
                return c
        raise KeyError(name)


def load_pack(path: str | Path) -> Pack:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Pack(
        world=data["world"],
        cards=[Card(**c) for c in data["cards"]],
        scenarios=data["scenarios"],
        provocations=data["provocations"],
        assignments=data["assignments"],
        boilerplate=data["boilerplate"],
        exemplars=data.get("exemplars", []),
    )


def card_system_prompt(card: Card, world: str) -> str:
    return (
        f"You are {card.name}, {card.persona}. Stay in character at all times. "
        f"Speak in a {card.style} manner, in the voice of {world}. "
        f"Never break character or mention being an AI."
    )


_NAME_STOPWORDS = {"the", "of", "a", "an", "le", "la", "du", "de", "von", "van"}


def name_tokens(name: str) -> set[str]:
    """Lowercased words of a character name, minus articles/particles."""
    return {w for w in name.lower().split() if w not in _NAME_STOPWORDS}


def distinctive_tokens(card: Card, cards: list[Card]) -> set[str]:
    """Name tokens unique to this card within the pack — the ones safe to key
    identity checks on. ("King Arthur" -> {king, arthur} unless another card
    also has "king" in their name.)"""
    others = set()
    for c in cards:
        if c.name != card.name:
            others |= name_tokens(c.name)
    return name_tokens(card.name) - others


def validate_pack(pack: Pack) -> list[str]:
    """Return a list of problems (empty = valid). Run me on your pack before
    burning teacher calls: `python -m personaforge.pack packs/mine.json`."""
    problems = []
    if not pack.world:
        problems.append("world is empty")
    for field_name in ("cards", "scenarios", "provocations", "assignments", "boilerplate"):
        if not getattr(pack, field_name):
            problems.append(f"{field_name} is empty")
    if len(pack.provocations) < 4 or len(pack.assignments) < 4:
        problems.append("need >=4 provocations and >=4 assignments so an eval "
                        "seed can be held out (every 4th is reserved)")
    names = [c.name for c in pack.cards]
    if len(set(names)) != len(names):
        problems.append("duplicate card names")
    for c in pack.cards:
        if not distinctive_tokens(c, pack.cards):
            problems.append(f"card {c.name!r} has no distinctive name token — "
                            "the wrong-character check and eval scoring need one")
    for tpl in pack.assignments:
        try:
            rendered = tpl.format(name="X", pron="them")
        except (KeyError, IndexError) as e:
            problems.append(f"assignment template {tpl!r} has a bad placeholder: {e}")
            continue
        if "{" in rendered:
            problems.append(f"assignment template {tpl!r} leaves unformatted braces")
    for ex in pack.exemplars:
        try:
            pack.card(ex["character"])
        except KeyError:
            problems.append(f"exemplar references unknown character {ex['character']!r}")
        if not all(len(t) == 2 for t in ex.get("turns", [])):
            problems.append(f"exemplar for {ex.get('character')} has a turn that "
                            "is not a [user, assistant] pair")
    return problems


def split_seeds(items: list[str], eval_every: int = 4) -> tuple[list[str], list[str]]:
    """Deterministic train/eval split of seed lists: every `eval_every`-th item
    is reserved for evaluation and never used to generate training data, so
    eval prompts are always unseen."""
    train = [x for i, x in enumerate(items) if i % eval_every != eval_every - 1]
    held = [x for i, x in enumerate(items) if i % eval_every == eval_every - 1]
    return train, held


def _cli() -> None:
    import sys
    pack = load_pack(sys.argv[1])
    problems = validate_pack(pack)
    if problems:
        print(f"INVALID pack ({len(problems)} problems):")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print(f"OK: {pack.world!r} — {len(pack.cards)} cards, "
          f"{len(pack.scenarios)} scenarios, {len(pack.provocations)} provocations, "
          f"{len(pack.assignments)} assignments, {len(pack.exemplars)} exemplars")


def exemplar_rows(pack: Pack) -> list[dict]:
    """Hand-written exemplar dialogues from the pack as SFT rows."""
    rows = []
    for ex in pack.exemplars:
        card = pack.card(ex["character"])
        messages = [{"role": "system", "content": card_system_prompt(card, pack.world)}]
        for user, assistant in ex["turns"]:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        row = {"messages": messages, "meta": {"source": "exemplar", "kind": "roleplay"}}
        validate_sft_row(row)
        rows.append(row)
    return rows
if __name__ == "__main__":
    _cli()
