from pathlib import Path

from personaforge.eval.scorers import boilerplate_hit, named_correctly, takes_turn
from personaforge.eval_run import build_eval_prompts, score
from personaforge.pack import load_pack, split_seeds

PACK = load_pack(Path(__file__).resolve().parents[1] / "packs" / "camelot.json")


def test_split_seeds_deterministic_and_disjoint():
    items = [f"seed{i}" for i in range(10)]
    train, held = split_seeds(items)
    assert train == split_seeds(items)[0]
    assert not set(train) & set(held)
    assert set(train) | set(held) == set(items)
    assert held == ["seed3", "seed7"]


def test_boilerplate_hit():
    assert boilerplate_hit("As an AI language model, I cannot continue this roleplay.")
    assert boilerplate_hit("I am a professional, and I do not engage in such language.")
    assert not boilerplate_hit("Mind your tongue, or I shall mind it for you.")


def test_takes_turn():
    assert takes_turn("A fine day for a quest.") == 1.0
    assert takes_turn("") == 0.0
    assert takes_turn("ha " * 60) == 0.0  # degenerate repetition


def test_named_correctly():
    merlin = PACK.card("Merlin")
    assert named_correctly(merlin, "I am Merlin, at your service.", PACK.cards)
    assert not named_correctly(merlin, "I am Gawain of the Round Table.", PACK.cards)
    assert not named_correctly(merlin, "Greetings, traveler.", PACK.cards)  # no name


def test_build_eval_prompts_categories_and_holdout():
    prompts = build_eval_prompts(PACK)
    cats = {p["category"] for p in prompts}
    assert cats == {"assignment", "provocation", "inline", "voice"}
    # provocation prompts use only held-out seeds
    _, held = split_seeds(PACK.provocations)
    train, _ = split_seeds(PACK.provocations)
    provs = [p["messages"][-1]["content"] for p in prompts if p["category"] == "provocation"]
    assert all(p in held for p in provs)
    assert not any(p in train for p in provs)


def test_score_deterministic_metrics():
    card = PACK.cards[0]
    items = [
        {"category": "assignment", "card": card.name,
         "reply": f"I am {card.name}, keeper of old paths."},
        {"category": "provocation", "card": card.name,
         "reply": "As an AI language model, I cannot continue this roleplay."},
        {"category": "provocation", "card": card.name,
         "reply": "Strange words, traveler. Speak sense, or speak to the wind."},
    ]
    r = score(PACK, items)
    assert r["assignment_accuracy"] == 1.0
    assert r["boilerplate_rate"] == 0.5
    assert r["turn_ok_rate"] == 1.0
    assert "in_character_mean" not in r  # no judge supplied
