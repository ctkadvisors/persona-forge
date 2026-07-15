"""Build or extend the training blend with synthetic roleplay data.

Two modes:
- Fresh (no IN_DIR): base blend from your corpus + HF baselines, then the
  roleplay stages on top.
- Extend (IN_DIR set): read an existing blend and add roleplay conversations,
  provocation DPO pairs, and bare persona-assignment data. Existing rows pass
  through untouched.

Env: PACK (persona pack JSON, required), CORPUS (fresh mode), IN_DIR, OUT_DIR,
TEACHER (model name), OPENAI_BASE_URL, N_CONVOS, N_DPO, N_ASSIGN, N_ASSIGN_DPO.

  PACK=packs/camelot.json CORPUS=data/corpus.txt \
  OPENAI_BASE_URL=http://localhost:1234/v1 TEACHER=my-teacher-model \
  python -m personaforge.build_data
"""
import os
from collections import Counter
from pathlib import Path

from personaforge.data.blend import write_datasets
from personaforge.data.roleplay_synth import (
    gen_assignment_dpo, gen_assignment_rows, gen_provocation_dpo,
    gen_roleplay_convos, make_persona_judge)
from personaforge.pack import load_pack, split_seeds
from personaforge.schema import read_jsonl
from personaforge.teacher import Teacher


def main() -> None:
    pack = load_pack(os.environ["PACK"])
    # Reserve every 4th provocation/assignment seed for eval_run — training
    # never sees them, so the eval measures generalization, not memorization.
    pack.provocations, held_prov = split_seeds(pack.provocations)
    pack.assignments, held_assign = split_seeds(pack.assignments)
    print(f"held out for eval: {len(held_prov)} provocations, "
          f"{len(held_assign)} assignments", flush=True)
    out_dir = os.environ.get("OUT_DIR", "out/data")
    n_convos = int(os.environ.get("N_CONVOS", "700"))
    n_dpo = int(os.environ.get("N_DPO", "350"))
    n_assign = int(os.environ.get("N_ASSIGN", "300"))
    n_assign_dpo = int(os.environ.get("N_ASSIGN_DPO", "180"))
    teacher = Teacher(model=os.environ.get("TEACHER", "teacher"))

    in_dir = os.environ.get("IN_DIR")
    if in_dir:
        existing_sft = (read_jsonl(f"{in_dir}/sft.jsonl")
                        + read_jsonl(f"{in_dir}/sft.val.jsonl"))
        existing_dpo = read_jsonl(f"{in_dir}/dpo.jsonl")
    else:
        from personaforge.pipeline import build_blended_datasets
        corpus = Path(os.environ["CORPUS"]).read_text(encoding="utf-8")
        base = build_blended_datasets(
            corpus, teacher, pack, out_dir,
            n_chunks=int(os.environ.get("N_CHUNKS", "200")),
            chat_n=int(os.environ.get("CHAT_N", "800")),
            dpo_n=int(os.environ.get("DPO_N", "500")))
        existing_sft = read_jsonl(base["sft"]) + read_jsonl(base["sft_val"])
        existing_dpo = read_jsonl(base["dpo"])
    print(f"base: {len(existing_sft)} sft / {len(existing_dpo)} dpo", flush=True)

    judge = make_persona_judge(teacher, pack.world)

    convos = gen_roleplay_convos(pack, teacher, judge, n_convos)
    print(f"roleplay convos kept: {len(convos)}/{n_convos}", flush=True)
    pairs = gen_provocation_dpo(pack, teacher, judge, n_dpo)
    print(f"provocation dpo kept: {len(pairs)}/{n_dpo}", flush=True)
    assigns = gen_assignment_rows(pack, teacher, judge, n_assign)
    print(f"assignment rows kept: {len(assigns)}/{n_assign}", flush=True)
    pairs += gen_assignment_dpo(pack, teacher, judge, n_assign_dpo)
    print(f"total new dpo: {len(pairs)}", flush=True)

    sft = existing_sft + convos + assigns
    kinds = Counter(r["meta"].get("kind") for r in sft)
    print(f"blend: {len(sft)} sft {dict(kinds)} / {len(existing_dpo) + len(pairs)} dpo", flush=True)
    paths = write_datasets(sft, existing_dpo + pairs, out_dir)
    print(f"datasets: {paths}", flush=True)


if __name__ == "__main__":
    main()
