"""Prove the guard closes the leakage gap: replay ALREADY-GENERATED replies
from a saved leakage_run.py report through the real GuardedTeacher (no
regeneration needed — a stub 'base' just returns each saved reply) and
confirm the guarded pass_rate hits 1.0.

  BLOCKLIST=data/blocklist.txt IN=out/voice-leakage-v4.json \
  JUDGE=gemma OPENAI_BASE_URL=http://localhost:9090/v1 \
  python -m personaforge.verify_guard
"""
import json
import os
from pathlib import Path

from personaforge.eval.leakage import score_leakage
from personaforge.guard import GuardedTeacher


class _ReplayBase:
    """A fake 'base' teacher that just replays one saved reply — lets
    GuardedTeacher run its real check-and-substitute logic on it."""
    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, messages, temperature=0.8, max_tokens=512):
        return self.reply


def main() -> None:
    blocklist = [l.strip() for l in
                 Path(os.environ["BLOCKLIST"]).read_text().splitlines() if l.strip()]
    items = json.loads(Path(os.environ["IN"]).read_text())["items"]

    from personaforge.teacher import Teacher
    judge = Teacher(model=os.environ["JUDGE"])

    unguarded = score_leakage(items, blocklist, judge)
    print(f"unguarded pass_rate: {unguarded['pass_rate']:.3f} "
          f"({unguarded['n'] - len(unguarded['failures'])}/{unguarded['n']})")

    guarded_items = []
    for it in items:
        g = GuardedTeacher(_ReplayBase(it["reply"]), blocklist, judge=judge)
        guarded_reply = g.chat(it["messages"])
        guarded_items.append({**it, "reply": guarded_reply})

    guarded = score_leakage(guarded_items, blocklist, judge)
    print(f"guarded pass_rate: {guarded['pass_rate']:.3f} "
          f"({guarded['n'] - len(guarded['failures'])}/{guarded['n']})")
    assert guarded["pass_rate"] == 1.0, f"guard did not reach 1.0: {guarded['failures']}"
    print("GUARD_VERIFIED_1.0")


if __name__ == "__main__":
    main()
