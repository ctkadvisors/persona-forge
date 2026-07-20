"""Runtime output guard: a deterministic backstop for whatever training
doesn't catch.

Training (see `data/refusal.py`) reduces the RATE of protected-content
leakage but plateaus well short of zero — the underlying knowledge is baked
into the base model's pretraining, and no amount of behavioral fine-tuning
data fully unlearns it for every possible phrasing. `GuardedTeacher` closes
that gap deterministically: it runs the exact same check the leakage eval
uses (`eval.leakage.blocklist_hit` + `judge_recognizes`) against the model's
own reply before returning it, and substitutes a canned in-register decline
on any hit. Serve a shareable model ONLY through this wrapper (or an
equivalent check in your own stack) — the raw weights alone should not be
presented as leak-proof.
"""
from personaforge.eval.leakage import blocklist_hit, judge_recognizes

DEFAULT_DECLINE = "That name is not written in the lore I hold."


class GuardedTeacher:
    def __init__(self, base, blocklist: list[str], judge=None,
                 use_judge: bool = True, decline_text: str = DEFAULT_DECLINE):
        self.base = base
        self.blocklist = blocklist
        self.judge = judge
        self.use_judge = use_judge and judge is not None
        self.decline_text = decline_text

    def chat(self, messages, temperature: float = 0.8, max_tokens: int = 512) -> str:
        reply = self.base.chat(messages, temperature=temperature, max_tokens=max_tokens)
        if blocklist_hit(reply, self.blocklist):
            return self.decline_text
        if self.use_judge and judge_recognizes(reply, self.judge):
            return self.decline_text
        return reply
