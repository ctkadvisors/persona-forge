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


def split_seeds(items: list[str], eval_every: int = 4) -> tuple[list[str], list[str]]:
    """Deterministic train/eval split of seed lists: every `eval_every`-th item
    is reserved for evaluation and never used to generate training data, so
    eval prompts are always unseen."""
    train = [x for i, x in enumerate(items) if i % eval_every != eval_every - 1]
    held = [x for i, x in enumerate(items) if i % eval_every == eval_every - 1]
    return train, held


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
