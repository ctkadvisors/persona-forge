import re

from personaforge.schema import validate_sft_row

_QA = re.compile(r"Q:\s*(.+?)\s*A:\s*(.+)", re.S)

_GEN_SYS = ("From the passage, write ONE question a reader might ask and a concise "
            "answer grounded ONLY in the passage. Format exactly:\nQ: <question>\nA: <answer>")


def gen_grounded_qa(passages, teacher, judge, per_passage: int = 1, temperature: float = 0.8):
    # per_passage > 1 with temperature > 0 yields multiple, varied (paraphrased)
    # QA pairs per passage — density that helps facts encode into the weights.
    out = []
    for passage in passages:
        for _ in range(per_passage):
            raw = teacher.chat([
                {"role": "system", "content": _GEN_SYS},
                {"role": "user", "content": passage},
            ], temperature=temperature)
            m = _QA.search(raw)
            if not m:
                continue
            q, a = m.group(1).strip(), m.group(2).strip()
            row = {
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "meta": {"source": "synth", "kind": "grounded_qa"},
            }
            if judge(row, passage) >= 0.7:
                validate_sft_row(row)
                out.append(row)
    return out


def make_judge(teacher):
    def judge(row, passage) -> float:
        verdict = teacher.chat([
            {"role": "system", "content":
             "Score 0..1 how fully the answer is supported by the passage. Reply with only the number."},
            {"role": "user", "content":
             f"PASSAGE:\n{passage}\n\nANSWER:\n{row['messages'][-1]['content']}"},
        ], temperature=0.0, max_tokens=8)
        try:
            return max(0.0, min(1.0, float(verdict.strip().split()[0])))
        except (ValueError, IndexError):
            return 0.0
    return judge
