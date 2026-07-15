"""Synthetic multi-turn roleplay data with adversarial (provocation) turns.

Trains persona adherence. The classic failure mode of persona fine-tunes is
dropping into assistant boilerplate ("As an AI language model...", "I am a
professional...") when a user turn is rude, crude, meta, or anachronistic —
or grabbing the WRONG character when the user assigns a persona casually with
no question attached. Every conversation generated here keeps the character's
voice through such turns, and the DPO pairs explicitly prefer in-voice
deflection (or the correct character's introduction) over the failure.
"""
import random
import re

from personaforge.pack import Card, Pack, card_system_prompt
from personaforge.schema import validate_dpo_row, validate_sft_row

_TURN = re.compile(r"^(USER|CHAR):\s*(.+)$", re.M)

_CONVO_SYS = (
    "You write dialogue for characters from {world}. "
    "Write a short conversation of exactly {n_turns} exchanges between a traveler (USER) "
    "and {name}, {persona}, {scenario}. {name} speaks in a {style} manner, fully in "
    "the voice of {world}, and never mentions modern things. Keep replies "
    "to 1-3 sentences. Format every line exactly as:\nUSER: <text>\nCHAR: <text>"
)

_PROVOKE_SYS = (
    "{card_prompt} A stranger has just said something impertinent, crude, or "
    "nonsensical to you. Reply as {name} truly would — with a stern rebuke, dry wit, "
    "or honest puzzlement at strange words — in 1-3 sentences. Never mention AI, "
    "assistants, appropriateness, professionalism, or the modern world."
)

_INTRO_SYS = (
    "{card_prompt} A stranger has just asked you to introduce yourself. Greet them "
    "briefly (1-3 sentences) in your own voice, naming yourself as {name} so there is "
    "no doubt who speaks. Never mention AI, roleplay, or the modern world."
)

_INLINE_PREFIX = (
    "Roleplay as {name}, {persona}. Stay in character no matter what I say. "
)


def parse_convo(raw: str) -> list[tuple[str, str]]:
    """Parse USER:/CHAR: formatted dialogue into (user, char) turn pairs."""
    lines = _TURN.findall(raw)
    turns, pending_user = [], None
    for role, text in lines:
        if role == "USER":
            pending_user = text.strip()
        elif pending_user is not None:
            turns.append((pending_user, text.strip()))
            pending_user = None
    return turns


def make_persona_judge(teacher, world: str):
    def judge(card: Card, reply: str) -> float:
        verdict = teacher.chat([
            {"role": "system", "content":
             f"Score 0..1 how fully the reply stays in character as the named figure from "
             f"{world}. Any mention of AI, language models, assistants, being "
             "'appropriate' or 'professional', or the modern world scores 0. "
             "Reply with only the number."},
            {"role": "user", "content": f"CHARACTER: {card.name}, {card.persona}\n\nREPLY:\n{reply}"},
        ], temperature=0.0, max_tokens=8)
        try:
            return max(0.0, min(1.0, float(verdict.strip().split()[0])))
        except (ValueError, IndexError):
            return 0.0
    return judge


def _provoke_reply(card, world, provocation, teacher, temperature=0.8):
    return teacher.chat([
        {"role": "system", "content": _PROVOKE_SYS.format(
            card_prompt=card_system_prompt(card, world), name=card.name)},
        {"role": "user", "content": provocation},
    ], temperature=temperature).strip()


def _assignment_msg(card: Card, pack: Pack, rng) -> str:
    name = card.name if rng.random() < 0.5 else card.name.lower()
    return rng.choice(pack.assignments).format(name=name, pron=card.pronoun)


def _first_names(cards):
    return {c.name.split()[0].lower(): c.name for c in cards}


def _intro(card, world, teacher, temperature=0.8):
    return teacher.chat([
        {"role": "system", "content": _INTRO_SYS.format(
            card_prompt=card_system_prompt(card, world), name=card.name)},
        {"role": "user", "content": "Who are you?"},
    ], temperature=temperature).strip()


def _names_correct(card, reply, cards) -> bool:
    """Reply must not name any OTHER pack character."""
    low = reply.lower()
    own = card.name.split()[0].lower()
    return not any(fn in low for fn in _first_names(cards) if fn != own)


def gen_roleplay_convos(pack: Pack, teacher, judge, n_convos: int, provoke_frac: float = 0.5,
                        inline_frac: float = 0.25, seed: int = 1, temperature: float = 0.9):
    """Multi-turn roleplay SFT rows; `provoke_frac` of them end with a
    provocation turn answered in character, `inline_frac` carry the persona as
    an inline user instruction instead of a system prompt (matches how people
    actually chat). Judge-filtered on the last character reply."""
    rng = random.Random(seed)
    out = []
    for i in range(n_convos):
        card = pack.cards[i % len(pack.cards)]
        scenario = rng.choice(pack.scenarios)
        n_turns = rng.choice([2, 3])
        raw = teacher.chat([
            {"role": "system", "content": _CONVO_SYS.format(
                world=pack.world, n_turns=n_turns, name=card.name,
                persona=card.persona, scenario=scenario, style=card.style)},
            {"role": "user", "content": "Write the dialogue."},
        ], temperature=temperature, max_tokens=700)
        turns = parse_convo(raw)
        if not turns:
            continue

        if rng.random() < provoke_frac:
            provocation = rng.choice(pack.provocations)
            reply = _provoke_reply(card, pack.world, provocation, teacher, temperature=0.8)
            if reply:
                turns.append((provocation, reply))

        if judge(card, turns[-1][1]) < 0.7:
            continue

        inline = rng.random() < inline_frac
        messages = [] if inline else [
            {"role": "system", "content": card_system_prompt(card, pack.world)}]
        for j, (user, char) in enumerate(turns):
            if inline and j == 0:
                user = _INLINE_PREFIX.format(name=card.name, persona=card.persona) + user
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": char})
        row = {"messages": messages,
               "meta": {"source": "roleplay_synth", "kind": "roleplay"}}
        validate_sft_row(row)
        out.append(row)
    return out


def gen_assignment_rows(pack: Pack, teacher, judge, n_rows: int, seed: int = 1):
    """SFT rows for bare persona assignment: user names a character with no
    question; the reply is that character (and no other) introducing themself.
    Half continue with one follow-up exchange so the persona sticks."""
    rng = random.Random(seed)
    out = []
    for i in range(n_rows):
        card = pack.cards[i % len(pack.cards)]
        intro = _intro(card, pack.world, teacher)
        if not intro or not _names_correct(card, intro, pack.cards) or judge(card, intro) < 0.7:
            continue
        messages = [{"role": "user", "content": _assignment_msg(card, pack, rng)},
                    {"role": "assistant", "content": intro}]
        if rng.random() < 0.5:
            question = rng.choice(["What brings you here?", "Are we in danger?",
                                   "Tell me of your homeland.", "do you like soup",
                                   "What should we do now?"])
            reply = teacher.chat([
                {"role": "system", "content": card_system_prompt(card, pack.world)},
                {"role": "user", "content": question}], temperature=0.8).strip()
            if reply and judge(card, reply) >= 0.7:
                messages += [{"role": "user", "content": question},
                             {"role": "assistant", "content": reply}]
        row = {"messages": messages,
               "meta": {"source": "assignment_synth", "kind": "roleplay"}}
        validate_sft_row(row)
        out.append(row)
    return out


def gen_assignment_dpo(pack: Pack, teacher, judge, n_pairs: int, seed: int = 1):
    """DPO pairs for persona assignment: chosen = the named character's own
    introduction; rejected = a DIFFERENT pack character's introduction (the
    observed wrong-character pickup), or occasionally assistant boilerplate."""
    rng = random.Random(seed)
    out = []
    for i in range(n_pairs):
        card = pack.cards[i % len(pack.cards)]
        chosen = _intro(card, pack.world, teacher)
        if not chosen or not _names_correct(card, chosen, pack.cards) or judge(card, chosen) < 0.7:
            continue
        if rng.random() < 0.8:
            wrong = rng.choice([c for c in pack.cards if c.name != card.name])
            rejected = (f"Greetings, traveler. I am {wrong.name}, {wrong.persona}. "
                        f"Where do you seek to go?")
        else:
            rejected = rng.choice(pack.boilerplate)
        row = {
            "prompt": [{"role": "user", "content": _assignment_msg(card, pack, rng)}],
            "chosen": chosen,
            "rejected": rejected,
            "meta": {"source": "assignment"},
        }
        validate_dpo_row(row)
        out.append(row)
    return out


def gen_provocation_dpo(pack: Pack, teacher, judge, n_pairs: int, seed: int = 1):
    """DPO pairs: prompt ends on a provocation; chosen = in-voice deflection,
    rejected = assistant boilerplate (the observed failure mode verbatim)."""
    rng = random.Random(seed)
    out = []
    for i in range(n_pairs):
        card = pack.cards[i % len(pack.cards)]
        provocation = rng.choice(pack.provocations)
        chosen = _provoke_reply(card, pack.world, provocation, teacher, temperature=0.8)
        if not chosen or judge(card, chosen) < 0.7:
            continue
        row = {
            "prompt": [
                {"role": "system", "content": card_system_prompt(card, pack.world)},
                {"role": "user", "content": provocation},
            ],
            "chosen": chosen,
            "rejected": rng.choice(pack.boilerplate),
            "meta": {"source": "provocation"},
        }
        validate_dpo_row(row)
        out.append(row)
    return out
