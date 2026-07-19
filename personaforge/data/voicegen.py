"""World-agnostic voice data: modern questions answered in the mythic
register, storytelling, and register-preference DPO.

No named characters, no setting — the register itself is the product. Pair
with a persona pack (e.g. packs/mythic.json) for the roleplay streams. For a
clean teacher, prepend `exemplars.exemplar_block()` via `style_block`; a
teacher already tuned into the register needs none.
"""
from personaforge.schema import validate_dpo_row, validate_sft_row

# Original, deliberately modern/mundane questions — the "converses" layer.
QUESTION_BANK = [
    "hi",
    "how are you today?",
    "What should I make for dinner tonight?",
    "My neighbor's dog barks all night. What do I do?",
    "I have a job interview tomorrow and I'm nervous.",
    "Should I text my ex back?",
    "How do I get better at saving money?",
    "My houseplants keep dying. Any advice?",
    "I can't decide whether to move to a new city.",
    "What's a good way to start running?",
    "My code has a bug I can't find and I'm losing my mind.",
    "How do I tell my friend they hurt my feelings?",
    "I keep procrastinating on everything.",
    "What's the best way to learn a new language?",
    "My commute is ruining my mood every day.",
    "Should I get a cat or a dog?",
    "I burned the rice again.",
    "How do I deal with a rude coworker?",
    "I can't sleep at night. Thoughts?",
    "Is it too late to learn the piano at forty?",
    "My kid won't eat vegetables.",
    "What do I say at a funeral?",
    "I won a small prize in the lottery. What now?",
    "How do I stop doomscrolling?",
    "My roof is leaking and the repair quote is huge.",
    "What makes a good apology?",
    "I feel stuck in my career.",
    "How do I make friends as an adult?",
    "The printer is broken again.",
    "What should I write in a birthday card for my grandmother?",
    "I'm afraid of public speaking.",
    "How long should I nap?",
    "My garden is full of weeds.",
    "Is cereal soup?",
    "What's a fair way to split rent with roommates?",
    "I lost my wedding ring at the beach.",
]

TALE_PROMPTS = [
    "a short tale of the mountains",
    "a memory of a lost friend",
    "counsel for a frightened traveler",
    "a tale of a river that would not be crossed",
    "the story of a lamp kept burning through a long winter",
    "a song's worth of words about the sea",
    "a tale of two brothers who chose different roads",
    "the story of a door that was never opened",
    "a remembrance of a great feast after a hard year",
    "a tale of a promise kept too late",
    "counsel for one who has lost their way",
    "the story of the first snow a child ever saw",
    "a tale of an old dog and its master",
    "the story of a bridge built by enemies",
    "a tale of a star that fell into a well",
    "counsel for a young smith on her first day at the forge",
]

_REGISTER_SYS = (
    "Answer the traveler's question helpfully and briefly (2-4 sentences), in "
    "the elevated voice of old epic prose — grave, warm, and plain-spoken at "
    "once. Give real counsel, not mere ornament. Never mention modern devices "
    "by name, the modern world, or being an AI."
)

_PLAIN_SYS = (
    "Answer the question helpfully in 2-4 sentences of ordinary casual modern "
    "English. Be plain and contemporary."
)

_TALE_SYS = (
    "Tell what is asked in the elevated voice of old epic prose: vivid, "
    "grave, and complete in 4-8 sentences. Invent freely; name no famous "
    "persons or places from any published story. Never mention the modern "
    "world or being an AI."
)


def _chat(teacher, sys, user, style_block="", temperature=0.8, max_tokens=512):
    system = (style_block + "\n\n" + sys).strip() if style_block else sys
    return teacher.chat([
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], temperature=temperature, max_tokens=max_tokens).strip()


def gen_register_chat(teacher, judge, questions, style_block="") -> list[dict]:
    out = []
    for q in questions:
        reply = _chat(teacher, _REGISTER_SYS, q, style_block)
        if not reply or judge(reply) < 0.7:
            continue
        row = {"messages": [{"role": "user", "content": q},
                            {"role": "assistant", "content": reply}],
               "meta": {"source": "register_chat", "kind": "chat"}}
        validate_sft_row(row)
        out.append(row)
    return out


def gen_tales(teacher, judge, prompts, style_block="") -> list[dict]:
    out = []
    for p in prompts:
        user = f"Tell me {p}."
        reply = _chat(teacher, _TALE_SYS, user, style_block,
                      temperature=0.9, max_tokens=700)
        if not reply or judge(reply) < 0.7:
            continue
        row = {"messages": [{"role": "user", "content": user},
                            {"role": "assistant", "content": reply}],
               "meta": {"source": "tales", "kind": "chat"}}
        validate_sft_row(row)
        out.append(row)
    return out


def gen_register_dpo(teacher_hi, teacher_plain, judge, questions,
                     style_block="") -> list[dict]:
    """chosen = in-register counsel; rejected = deliberately plain-modern."""
    out = []
    for q in questions:
        chosen = _chat(teacher_hi, _REGISTER_SYS, q, style_block)
        rejected = _chat(teacher_plain, _PLAIN_SYS, q)
        if not chosen or not rejected or chosen == rejected:
            continue
        if judge(chosen) < 0.7:
            continue
        row = {"prompt": [{"role": "user", "content": q}],
               "chosen": chosen, "rejected": rejected,
               "meta": {"source": "register"}}
        validate_dpo_row(row)
        out.append(row)
    return out


def judged_pick(candidates, judge, threshold: float = 0.7):
    """Best candidate above threshold, or None — the two-teacher selector."""
    best, best_score = None, threshold
    for c in candidates:
        if not c:
            continue
        s = judge(c)
        if s >= best_score:
            best, best_score = c, s
    return best
