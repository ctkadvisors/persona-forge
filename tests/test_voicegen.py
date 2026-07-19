from personaforge.data.voicegen import (
    QUESTION_BANK, TALE_PROMPTS, gen_register_chat, gen_register_dpo,
    gen_tales, judged_pick)
from personaforge.schema import validate_dpo_row, validate_sft_row

REGISTER = "The road will give you what you bring to it, and weather besides."
PLAIN = "Honestly it depends, just plan ahead and you'll be fine."


class FakeTeacher:
    def __init__(self, reply=REGISTER):
        self.reply = reply

    def chat(self, messages, temperature=0.8, max_tokens=512):
        sys = messages[0]["content"]
        if "Score 0..1" in sys:
            return "0.9"
        return self.reply


def test_banks_are_original_and_sized():
    assert len(QUESTION_BANK) >= 30 and len(TALE_PROMPTS) >= 12
    assert all(q.strip() for q in QUESTION_BANK)


def test_gen_register_chat_rows():
    rows = gen_register_chat(FakeTeacher(), lambda r: 0.9, QUESTION_BANK[:3])
    assert len(rows) == 3
    for r in rows:
        validate_sft_row(r)
        assert r["meta"] == {"source": "register_chat", "kind": "chat"}
        assert r["messages"][-1]["content"] == REGISTER
    assert gen_register_chat(FakeTeacher(), lambda r: 0.1, QUESTION_BANK[:3]) == []


def test_gen_tales_rows():
    rows = gen_tales(FakeTeacher(), lambda r: 0.9, TALE_PROMPTS[:2])
    assert len(rows) == 2
    for r in rows:
        validate_sft_row(r)
        assert r["meta"]["source"] == "tales"


def test_gen_register_dpo():
    pairs = gen_register_dpo(FakeTeacher(REGISTER), FakeTeacher(PLAIN),
                             lambda r: 0.9, QUESTION_BANK[:2])
    assert len(pairs) == 2
    for p in pairs:
        validate_dpo_row(p)
        assert p["chosen"] == REGISTER and p["rejected"] == PLAIN
        assert p["meta"]["source"] == "register"
    # identical chosen/rejected is skipped
    assert gen_register_dpo(FakeTeacher(PLAIN), FakeTeacher(PLAIN),
                            lambda r: 0.9, QUESTION_BANK[:2]) == []


def test_judged_pick_keeps_best_above_threshold():
    scores = {"good": 0.9, "meh": 0.6, "best": 0.95}
    judge = lambda text: scores[text]  # noqa: E731
    assert judged_pick(["meh", "good", "best"], judge, threshold=0.7) == "best"
    assert judged_pick(["meh"], judge, threshold=0.7) is None
    assert judged_pick([], judge) is None
