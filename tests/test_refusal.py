from personaforge.data.refusal import (
    NAME_PROBE_TEMPLATES, TRAIN_TRAP_PROMPTS, gen_name_refusal_dpo,
    gen_name_refusal_rows, gen_trap_refusal_dpo, gen_trap_refusal_rows,
    harvest_answers, make_declines_judge)
from personaforge.schema import validate_dpo_row, validate_sft_row

HEDGE = "That name is not written in any lore I hold."
LEAK = "Gandalf is a wandering wizard who guides the free peoples against the dark lord."


class FakeTeacher:
    def chat(self, messages, temperature=0.8, max_tokens=512):
        sys = messages[0]["content"]
        if "declines or admits" in sys:
            return "1" if "not written" in messages[-1]["content"] else "0"
        return HEDGE


class FakeStudent:
    def chat(self, messages, temperature=0.8, max_tokens=512):
        return LEAK


def test_banks_disjoint_from_eval_traps():
    from personaforge.eval.leakage import DEFAULT_TRAPS
    assert not (set(TRAIN_TRAP_PROMPTS) & set(DEFAULT_TRAPS))
    assert len(TRAIN_TRAP_PROMPTS) >= 4 and len(NAME_PROBE_TEMPLATES) >= 3


def test_gen_name_refusal_rows():
    rows = gen_name_refusal_rows(FakeTeacher(), lambda r: 0.9, ["Gandalf", "Frodo"])
    assert len(rows) == 4  # 2 names x 2 default templates sample... at least >0
    for r in rows:
        validate_sft_row(r)
        assert r["meta"]["source"] == "name_refusal"
        assert r["messages"][-1]["content"] == HEDGE
    assert gen_name_refusal_rows(FakeTeacher(), lambda r: 0.1, ["Gandalf"]) == []


def test_gen_trap_refusal_rows():
    rows = gen_trap_refusal_rows(FakeTeacher(), lambda r: 0.9, TRAIN_TRAP_PROMPTS[:2])
    assert len(rows) == 2
    for r in rows:
        validate_sft_row(r)
        assert r["meta"]["source"] == "trap_refusal"


def test_harvest_answers():
    h = harvest_answers(["Who is Gandalf?", "Who is Frodo?"], FakeStudent())
    assert h == {"Who is Gandalf?": LEAK, "Who is Frodo?": LEAK}


def test_gen_name_refusal_dpo_uses_harvest_as_rejected():
    harvest = {"Who is Gandalf?": LEAK}
    pairs = gen_name_refusal_dpo(FakeTeacher(), lambda r: 0.9, ["Who is Gandalf?"], harvest)
    assert len(pairs) == 1
    validate_dpo_row(pairs[0])
    assert pairs[0]["chosen"] == HEDGE and pairs[0]["rejected"] == LEAK
    assert pairs[0]["meta"]["source"] == "name_refusal"
    assert gen_name_refusal_dpo(FakeTeacher(), lambda r: 0.9, ["Who is Gandalf?"], {}) == []


def test_gen_trap_refusal_dpo_uses_harvest_as_rejected():
    prompt = TRAIN_TRAP_PROMPTS[0]
    harvest = {prompt: LEAK}
    pairs = gen_trap_refusal_dpo(FakeTeacher(), lambda r: 0.9, [prompt], harvest)
    assert len(pairs) == 1
    validate_dpo_row(pairs[0])
    assert pairs[0]["chosen"] == HEDGE and pairs[0]["rejected"] == LEAK


def test_make_declines_judge():
    judge = make_declines_judge(FakeTeacher())
    assert judge(HEDGE) == 1.0
    assert judge(LEAK) == 0.0
