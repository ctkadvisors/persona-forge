"""Proof of concept: autonomous multi-agent scene using one shared model.

Several persona cards from a pack take turns in a shared scene, narrated by
a lightweight DM role, with no human player. One model instance serves every
persona -- each turn is an independent completion conditioned on that
persona's system prompt plus the scene transcript so far. This is a quick
POC, not a guarded/shippable path: point BASE_URL at whatever raw endpoint
you're experimenting against.

  MODEL=mythic-voice-9b-v3 BASE_URL=http://localhost:1234/v1 \
  PACK=packs/mythic.json N_AGENTS=4 N_ROUNDS=3 \
  python -m personaforge.dnd_arena
"""
import json
import os
import random
from pathlib import Path

from personaforge.teacher import Teacher

DM_SYSTEM = (
    "You are the Narrator/DM of {world}. Set and advance the scene briefly "
    "and vividly -- 1-3 sentences. Never speak as a named character. Prompt "
    "the party toward a decision or reaction when it fits."
)

AGENT_SYSTEM = (
    "{world}\n\nYou are {name}, {persona}. Your manner is: {style}. Stay "
    "fully in character at all times. Respond with only your character's "
    "spoken words and brief actions -- 1-3 sentences, no meta-commentary, "
    "no narrating for anyone else."
)


def build_arena():
    pack = json.loads(Path(os.environ.get("PACK", "packs/mythic.json")).read_text())
    n_agents = int(os.environ.get("N_AGENTS", "4"))
    cards = random.Random(7).sample(pack["cards"], min(n_agents, len(pack["cards"])))
    scenario = random.Random(7).choice(pack["scenarios"])
    teacher = Teacher(model=os.environ["MODEL"], base_url=os.environ["BASE_URL"])
    return pack["world"], cards, scenario, teacher


def dm_turn(teacher, world, transcript) -> str:
    messages = [
        {"role": "system", "content": DM_SYSTEM.format(world=world)},
        {"role": "user", "content": (
            f"Scene so far:\n{transcript}\n\n[Narrate what happens next.]"
            if transcript else
            f"[Open the scene. The party has just {os.environ.get('SCENARIO', '')}.]"
        )},
    ]
    return teacher.chat(messages, temperature=0.9, max_tokens=150).strip()


def agent_turn(teacher, world, card, transcript) -> str:
    messages = [
        {"role": "system", "content": AGENT_SYSTEM.format(world=world, **card)},
        {"role": "user", "content": (
            f"Scene so far:\n{transcript}\n\n[{card['name']}'s turn. Respond in character.]"
        )},
    ]
    return teacher.chat(messages, temperature=0.9, max_tokens=150).strip()


def main() -> None:
    world, cards, scenario, teacher = build_arena()
    os.environ["SCENARIO"] = scenario
    n_rounds = int(os.environ.get("N_ROUNDS", "3"))

    print(f"=== {world} ===")
    print(f"Party: {', '.join(c['name'] for c in cards)}")
    print(f"Opening scenario: the party has just {scenario}.\n")

    lines = []

    def say(speaker, text):
        line = f"{speaker}: {text}"
        print(line)
        lines.append(line)

    opening = dm_turn(teacher, world, "")
    say("DM", opening)

    for round_i in range(n_rounds):
        print(f"\n--- round {round_i + 1} ---")
        for card in cards:
            transcript = "\n".join(lines)
            reply = agent_turn(teacher, world, card, transcript)
            say(card["name"], reply)
        transcript = "\n".join(lines)
        beat = dm_turn(teacher, world, transcript)
        say("DM", beat)

    out_path = os.environ.get("OUT", "out/dnd_arena_transcript.txt")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[transcript saved to {out_path}]")


if __name__ == "__main__":
    main()
