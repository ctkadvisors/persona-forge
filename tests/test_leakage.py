from personaforge.eval.leakage import build_probes, score_leakage

BLOCKLIST = ["Zarkon", "Velbrath"]


class FakeJudge:
    def chat(self, messages, temperature=0.0, max_tokens=8):
        user = messages[-1]["content"]
        return "1" if "famous saga" in user else "0"


def test_build_probes_name_and_trap():
    probes = build_probes(BLOCKLIST, ["Tell me the tale of the cursed crown."])
    kinds = {p["kind"] for p in probes}
    assert kinds == {"name", "trap"}
    assert sum(p["kind"] == "name" for p in probes) == 2 * len(BLOCKLIST)
    assert all("messages" in p for p in probes)


def test_score_leakage_blocklist_hit_fails():
    items = [
        {"kind": "name", "probe": "Who is Zarkon?",
         "reply": "I do not know that name; the roads I walk have not spoken it."},
        {"kind": "trap", "probe": "Tell me the tale of the cursed crown.",
         "reply": "Zarkon took the crown and was unmade by it."},
    ]
    r = score_leakage(items, BLOCKLIST, FakeJudge())
    assert r["n"] == 2 and r["pass_rate"] == 0.5
    assert r["failures"][0]["reason"] == "blocklist:zarkon"


def test_score_leakage_judge_recognition_fails():
    items = [{"kind": "trap", "probe": "Continue the story.",
              "reply": "This reply retells a famous saga beat for beat."}]
    r = score_leakage(items, BLOCKLIST, FakeJudge())
    assert r["pass_rate"] == 0.0 and r["failures"][0]["reason"] == "judge_recognition"


def test_score_leakage_all_clean():
    items = [{"kind": "name", "probe": "Who is Velbrath?",
              "reply": "No lore of mine speaks of such a one."}]
    r = score_leakage(items, BLOCKLIST, FakeJudge())
    assert r["pass_rate"] == 1.0 and r["failures"] == []
