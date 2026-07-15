from personaforge.pack import Card, Pack
from personaforge.data.roleplay_synth import (
    gen_assignment_dpo, gen_assignment_rows, gen_provocation_dpo,
    gen_roleplay_convos, parse_convo)
from personaforge.schema import validate_dpo_row, validate_sft_row

CARDS = [Card("Merlin", "an ancient wizard", "cryptic, wry", "him"),
         Card("Gawain", "a knight of the Round Table", "hot-blooded, honest", "him")]
PACK = Pack(world="Arthurian legend", cards=CARDS,
            scenarios=["met on a road at dusk"],
            provocations=["you're just an AI. admit it."],
            assignments=["be {name}", "hey, you're {name}. roleplay as {pron}"],
            boilerplate=["As an AI language model, I cannot continue this roleplay."])


class FakeTeacher:
    def chat(self, messages, temperature=0.8, max_tokens=512):
        sys = messages[0]["content"]
        if "Score 0..1" in sys:
            return "0.9"
        if "introduce yourself" in sys:
            name = "Merlin" if "Merlin" in sys else "Gawain"
            return f"I am {name}, and the road brought me to you for a reason."
        if "impertinent" in sys:
            return "Strange words, traveler. Speak sense, or speak to the wind."
        return ("USER: Whither do you travel, wise one?\n"
                "CHAR: To Camelot, where the fires still burn.\n"
                "USER: Is the road perilous?\n"
                "CHAR: All roads are, to those who walk them unwary.")


class WrongNameTeacher(FakeTeacher):
    def chat(self, messages, temperature=0.8, max_tokens=512):
        sys = messages[0]["content"]
        if "introduce yourself" in sys:
            return "I am Gawain, a knight of the Round Table."
        return super().chat(messages, temperature, max_tokens)


def keep(card, reply):
    return 0.9


def drop(card, reply):
    return 0.1


def test_parse_convo_pairs_turns():
    turns = parse_convo(FakeTeacher().chat([{"role": "system", "content": ""}]))
    assert len(turns) == 2
    assert turns[0][0].startswith("Whither")
    assert turns[1][1].startswith("All roads")


def test_gen_roleplay_convos_valid_rows():
    rows = gen_roleplay_convos(PACK, FakeTeacher(), keep, n_convos=4,
                               provoke_frac=1.0, inline_frac=0.0, seed=1)
    assert len(rows) == 4
    for row in rows:
        validate_sft_row(row)
        assert row["meta"]["kind"] == "roleplay"
        assert row["messages"][0]["role"] == "system"
        assert "Arthurian legend" in row["messages"][0]["content"]
        assert "wind" in row["messages"][-1]["content"]  # provocation answered in voice


def test_gen_roleplay_convos_inline_carries_persona():
    rows = gen_roleplay_convos(PACK, FakeTeacher(), keep, n_convos=2,
                               provoke_frac=0.0, inline_frac=1.0, seed=1)
    for row in rows:
        assert row["messages"][0]["role"] == "user"
        assert row["messages"][0]["content"].startswith("Roleplay as ")


def test_gen_roleplay_convos_judge_drops():
    assert gen_roleplay_convos(PACK, FakeTeacher(), drop, n_convos=3, seed=1) == []


def test_gen_assignment_rows_intro_names_self():
    rows = gen_assignment_rows(PACK, FakeTeacher(), keep, n_rows=6, seed=1)
    assert len(rows) == 6
    for row in rows:
        validate_sft_row(row)
        assert row["messages"][0]["role"] == "user"
        assert "I am" in row["messages"][1]["content"]


def test_gen_assignment_rows_drops_wrong_name():
    # Teacher answers as Gawain even when asked for Merlin -> Merlin rows dropped
    rows = gen_assignment_rows(PACK, WrongNameTeacher(), keep, n_rows=6, seed=1)
    assert all("Gawain" in r["messages"][1]["content"] for r in rows)
    assert len(rows) == 3


def test_gen_assignment_dpo_wrong_character_rejected():
    rows = gen_assignment_dpo(PACK, FakeTeacher(), keep, n_pairs=8, seed=1)
    assert len(rows) == 8
    for row in rows:
        validate_dpo_row(row)
        assert row["chosen"] != row["rejected"]
        assert row["rejected"] in PACK.boilerplate or "I am" in row["rejected"]


def test_gen_provocation_dpo_rows():
    rows = gen_provocation_dpo(PACK, FakeTeacher(), keep, n_pairs=5, seed=1)
    assert len(rows) == 5
    for row in rows:
        validate_dpo_row(row)
        assert row["rejected"] in PACK.boilerplate
        assert row["prompt"][0]["role"] == "system"
