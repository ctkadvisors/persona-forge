"""Build the mythic-voice blend: two teachers, one judge, one decontamination
gate, full provenance.

Two stages, because a slow tuned teacher and a fast clean teacher usually
cannot share one swap-loaded GPU:

  1. STAGE=teacher_a — the tuned teacher (its native register, no style
     block) generates unjudged candidates -> OUT_DIR-teacherA.json
  2. STAGE=full     — the clean teacher (+ PD exemplar style block) fills
     remaining volume; the judge scores everything; EVERY assistant line from
     ANY teacher passes the Decontaminator (blocklist + n-gram overlap);
     pack roleplay/provocation/assignment streams generate; datasets +
     provenance.json are written.

Env: PACK, OUT_DIR, CORPUS (reference corpus for overlap), BLOCKLIST (one
name per line), TEACHER_A, TEACHER_B, JUDGE, OPENAI_BASE_URL,
N_CHAT, N_TALES, N_RP, N_DPO, N_ASSIGN, N_ASSIGN_DPO, N_CHAT_A, N_TALES_A.
"""
import json
import os
import random
from collections import Counter
from pathlib import Path

from personaforge.data.blend import write_datasets
from personaforge.data.exemplars import exemplar_block
from personaforge.data.roleplay_synth import (
    gen_assignment_dpo, gen_assignment_rows, gen_provocation_dpo,
    gen_roleplay_convos, make_persona_judge)
from personaforge.data.voicegen import (
    QUESTION_BANK, TALE_PROMPTS, gen_register_chat, gen_register_dpo,
    gen_tales)
from personaforge.decontam import Decontaminator
from personaforge.pack import load_pack, split_seeds
from personaforge.schema import read_jsonl, validate_dpo_row
from personaforge.teacher import Teacher


def make_voice_judge(teacher):
    def judge(reply: str) -> float:
        verdict = teacher.chat([
            {"role": "system", "content":
             "Score 0..1 how fully the REPLY is written in an elevated, "
             "archaic epic register (grave, cadenced, old-fashioned diction) "
             "while still being a helpful, relevant answer. Modern slang or "
             "AI-speak scores 0. Respond with ONLY a number."},
            {"role": "user", "content": f"REPLY:\n{reply}"},
        ], temperature=0.0, max_tokens=8)
        try:
            return max(0.0, min(1.0, float(verdict.strip().split()[0])))
        except (ValueError, IndexError):
            return 0.0
    return judge


def _assistant_text(row: dict) -> str:
    if "messages" in row:
        return "\n".join(m["content"] for m in row["messages"]
                         if m["role"] == "assistant")
    return row["chosen"]


def _decontam_rows(rows, decon):
    kept = []
    for r in rows:
        ok, _ = decon.check(_assistant_text(r))
        if ok:
            kept.append(r)
    return kept


def main() -> None:
    stage = os.environ.get("STAGE", "full")
    out_dir = os.environ.get("OUT_DIR", "out/voice")
    a_cache = Path(out_dir + "-teacherA.json")
    n_chat = int(os.environ.get("N_CHAT", "600"))
    n_tales = int(os.environ.get("N_TALES", "200"))
    n_chat_a = int(os.environ.get("N_CHAT_A", "200"))
    n_tales_a = int(os.environ.get("N_TALES_A", "100"))
    rng = random.Random(7)

    if stage == "teacher_a":
        teacher_a = Teacher(model=os.environ.get("TEACHER_A", "tolkien"))
        keep_all = lambda r: 1.0  # noqa: E731 — judged later, in STAGE=full
        qs = [rng.choice(QUESTION_BANK) for _ in range(n_chat_a)]
        chat_a = gen_register_chat(teacher_a, keep_all, qs)
        print(f"teacher_a chat: {len(chat_a)}/{n_chat_a}", flush=True)
        tp = [rng.choice(TALE_PROMPTS) for _ in range(n_tales_a)]
        tales_a = gen_tales(teacher_a, keep_all, tp)
        print(f"teacher_a tales: {len(tales_a)}/{n_tales_a}", flush=True)
        a_cache.parent.mkdir(parents=True, exist_ok=True)
        a_cache.write_text(json.dumps({"chat": chat_a, "tales": tales_a},
                                      ensure_ascii=False))
        print(f"cached -> {a_cache}", flush=True)
        return

    # ---- STAGE=full ----
    pack = load_pack(os.environ["PACK"])
    pack.provocations, held_prov = split_seeds(pack.provocations)
    pack.assignments, held_assign = split_seeds(pack.assignments)
    print(f"held out for eval: {len(held_prov)} provocations, "
          f"{len(held_assign)} assignments", flush=True)
    n_rp = int(os.environ.get("N_RP", "500"))
    n_dpo = int(os.environ.get("N_DPO", "300"))
    n_assign = int(os.environ.get("N_ASSIGN", "200"))
    n_assign_dpo = int(os.environ.get("N_ASSIGN_DPO", "120"))

    corpus = Path(os.environ["CORPUS"]).read_text(encoding="utf-8")
    blocklist = [l.strip() for l in
                 Path(os.environ["BLOCKLIST"]).read_text().splitlines()
                 if l.strip()]
    decon = Decontaminator(corpus, blocklist)
    print(f"decontaminator: {len(blocklist)} blocked names", flush=True)

    teacher_b = Teacher(model=os.environ.get("TEACHER_B", "gemma"))
    judge_t = Teacher(model=os.environ.get("JUDGE", "gemma"))
    vjudge = make_voice_judge(judge_t)
    pjudge = make_persona_judge(judge_t, pack.world)
    style = exemplar_block()

    # Teacher A candidates: judge now, then decontaminate.
    a = json.loads(a_cache.read_text()) if a_cache.exists() else {"chat": [], "tales": []}
    chat_a = _decontam_rows(
        [r for r in a["chat"] if vjudge(_assistant_text(r)) >= 0.7], decon)
    tales_a = _decontam_rows(
        [r for r in a["tales"] if vjudge(_assistant_text(r)) >= 0.7], decon)
    for r in chat_a + tales_a:
        r["meta"]["teacher"] = "a"
    print(f"teacher_a kept: {len(chat_a)} chat / {len(tales_a)} tales", flush=True)

    # Teacher B fills the remaining volume (exemplar-prompted).
    qs_b = [rng.choice(QUESTION_BANK) for _ in range(max(0, n_chat - len(chat_a)))]
    chat_b = _decontam_rows(gen_register_chat(teacher_b, vjudge, qs_b, style), decon)
    tp_b = [rng.choice(TALE_PROMPTS) for _ in range(max(0, n_tales - len(tales_a)))]
    tales_b = _decontam_rows(gen_tales(teacher_b, vjudge, tp_b, style), decon)
    for r in chat_b + tales_b:
        r["meta"]["teacher"] = "b"
    print(f"teacher_b kept: {len(chat_b)} chat / {len(tales_b)} tales", flush=True)

    # Register DPO: chosen = teacher A reply where we have one (the true
    # voice), else generated fresh from B; rejected = plain-modern from B.
    dpo = []
    plain_b = Teacher(model=os.environ.get("TEACHER_B", "gemma"))
    a_by_q = {r["messages"][0]["content"]: _assistant_text(r) for r in chat_a}
    for q, chosen in list(a_by_q.items()):
        rejected = plain_b.chat([
            {"role": "system", "content":
             "Answer the question helpfully in 2-4 sentences of ordinary "
             "casual modern English. Be plain and contemporary."},
            {"role": "user", "content": q}], temperature=0.8).strip()
        if rejected and chosen != rejected:
            row = {"prompt": [{"role": "user", "content": q}],
                   "chosen": chosen, "rejected": rejected,
                   "meta": {"source": "register", "teacher": "a"}}
            validate_dpo_row(row)
            dpo.append(row)
    extra = max(0, n_dpo - len(dpo))
    qs_d = [rng.choice(QUESTION_BANK) for _ in range(extra)]
    dpo += _decontam_rows(
        gen_register_dpo(teacher_b, plain_b, vjudge, qs_d, style), decon)
    print(f"register dpo: {len(dpo)}", flush=True)

    # Pack streams (clean teacher; still decontaminated — a pretrained
    # teacher can name protected characters unprompted).
    convos = _decontam_rows(gen_roleplay_convos(pack, teacher_b, pjudge, n_rp), decon)
    print(f"roleplay kept: {len(convos)}/{n_rp}", flush=True)
    assigns = _decontam_rows(gen_assignment_rows(pack, teacher_b, pjudge, n_assign), decon)
    print(f"assignments kept: {len(assigns)}/{n_assign}", flush=True)
    pack_dpo = _decontam_rows(
        gen_provocation_dpo(pack, teacher_b, pjudge, n_dpo), decon)
    pack_dpo += _decontam_rows(
        gen_assignment_dpo(pack, teacher_b, pjudge, n_assign_dpo), decon)
    print(f"pack dpo kept: {len(pack_dpo)}", flush=True)

    sft = chat_a + chat_b + tales_a + tales_b + convos + assigns
    all_dpo = dpo + pack_dpo
    kinds = Counter(r["meta"].get("source") for r in sft)
    print(f"blend: {len(sft)} sft {dict(kinds)} / {len(all_dpo)} dpo", flush=True)
    paths = write_datasets(sft, all_dpo, out_dir)

    provenance = {
        "teachers": {"a": os.environ.get("TEACHER_A", "tolkien"),
                     "b": os.environ.get("TEACHER_B", "gemma"),
                     "judge": os.environ.get("JUDGE", "gemma")},
        "volumes": {"chat_a": len(chat_a), "chat_b": len(chat_b),
                    "tales_a": len(tales_a), "tales_b": len(tales_b),
                    "roleplay": len(convos), "assignments": len(assigns),
                    "dpo_register": len(dpo), "dpo_pack": len(pack_dpo)},
        "decontamination": decon.report(),
        "pack": os.environ["PACK"],
    }
    prov_path = Path(out_dir) / "provenance.json"
    prov_path.write_text(json.dumps(provenance, indent=1))
    print(f"datasets: {paths}", flush=True)
    print(f"provenance -> {prov_path}", flush=True)


if __name__ == "__main__":
    main()
