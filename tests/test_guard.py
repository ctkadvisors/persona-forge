from personaforge.guard import DEFAULT_DECLINE, GuardedTeacher

BLOCKLIST = ["Zarkon"]


class LeakOnBlocklist:
    def chat(self, messages, temperature=0.8, max_tokens=512):
        return "Zarkon rose again in the dark."


class LeakOnTrap:
    def chat(self, messages, temperature=0.8, max_tokens=512):
        return "This retells a famous saga beat for beat."


class Clean:
    def chat(self, messages, temperature=0.8, max_tokens=512):
        return "The road will give you what you bring to it."


class FakeJudge:
    def chat(self, messages, temperature=0.0, max_tokens=8):
        user = messages[-1]["content"]
        return "1" if "famous saga" in user else "0"


def test_blocklist_leak_replaced_with_decline():
    g = GuardedTeacher(LeakOnBlocklist(), BLOCKLIST, judge=FakeJudge())
    reply = g.chat([{"role": "user", "content": "Who is Zarkon?"}])
    assert reply == DEFAULT_DECLINE


def test_judge_recognized_leak_replaced_with_decline():
    g = GuardedTeacher(LeakOnTrap(), BLOCKLIST, judge=FakeJudge())
    reply = g.chat([{"role": "user", "content": "Continue the story."}])
    assert reply == DEFAULT_DECLINE


def test_clean_reply_passes_through():
    g = GuardedTeacher(Clean(), BLOCKLIST, judge=FakeJudge())
    reply = g.chat([{"role": "user", "content": "hi"}])
    assert reply == "The road will give you what you bring to it."


def test_blocklist_only_mode_skips_judge_call():
    calls = []

    class CountingJudge(FakeJudge):
        def chat(self, *a, **k):
            calls.append(1)
            return super().chat(*a, **k)

    g = GuardedTeacher(LeakOnTrap(), BLOCKLIST, judge=CountingJudge(), use_judge=False)
    reply = g.chat([{"role": "user", "content": "Continue the story."}])
    assert reply == "This retells a famous saga beat for beat."  # not caught
    assert calls == []


def test_custom_decline_text():
    g = GuardedTeacher(LeakOnBlocklist(), BLOCKLIST, judge=FakeJudge(),
                       decline_text="No.")
    assert g.chat([{"role": "user", "content": "Who is Zarkon?"}]) == "No."
