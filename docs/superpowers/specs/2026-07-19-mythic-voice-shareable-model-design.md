# Design: Mythic-Voice Shareable Model

**Date:** 2026-07-19
**Status:** Approved (user, in-session)
**Home:** persona-forge (pipeline, public) + private-side runs (teacher, corpus, blocklist)

## Goal

A **legally shareable** model that speaks in an elevated archaic-mythic English
register, carries user-supplied personas, resists provocation, and tells
stories — with **no baked-in copyrighted world**. Staged release: trusted beta
tester first, then public HF release with a provenance report.

## Legal posture (engineering summary, not legal advice)

1. **Style is not copyrightable.** The register (archaic diction, inversion,
   epic cadence) is free to learn and to sell. The public-domain sources that
   formed the style (Malory, Morris, Eddas, Kalevala, KJV, Eddison) are the
   documented lineage.
2. **Names and expression are protectable.** The pipeline mechanically
   excludes both: a proper-noun blocklist (the "TSR balrog→balor move",
   enforced by code) and an n-gram overlap gate against the private corpus
   (catches memorized expression — the hazard unique to trained weights).
3. **No claimed association.** The artifact's name, model card, and examples
   never reference the inspiring author, estate, or trademarks. Positioning:
   "archaic mythic register in the public-domain northern-European epic
   tradition."
4. **Provenance report ships with the model**: teacher mix, filter pass/drop
   counts, leakage-eval results. Defensible, not hopeful.

Decisions made during brainstorming: audience = **both, staged** (public rigor,
beta first); world = **world-agnostic voice model** (users bring personas, as
proven by the beta tester's D&D character over a system prompt); teachers =
**both, judged** (private tuned 27B for the true voice + clean teacher with PD
exemplars; judge keeps the best; tuned-teacher samples must survive
decontamination).

## Architecture

**Public (persona-forge)** — all generic, no private data:

- `personaforge/decontam.py` — `Decontaminator(corpus_text, blocklist,
  n=8)`: `check(text) -> (ok, reason)`; n-gram overlap gate + word-boundary
  blocklist; `report()` with pass/drop counts by reason. Corpus and blocklist
  are runtime inputs — the module never embeds them.
- `personaforge/data/exemplars.py` — short public-domain style exemplars
  (Malory, Morris, KJV cadence) used to prompt the clean teacher.
- `personaforge/data/voicegen.py` — world-agnostic generators:
  - `gen_register_chat(...)` — modern-question → in-register answer rows
    (the "converses without genealogy-barfing" layer).
  - `gen_tales(...)` — storytelling rows from generic prompts (a tale of
    the mountains, counsel for the fearful, a memory of a lost friend).
  - Roleplay/provocation/assignment reuse the existing pack-driven
    generators with a new **`packs/mythic.json`**: archetype cards with
    original invented names (ours), generic scenarios, the standard
    provocation set.
  - `gen_judged(...)` — two-teacher wrapper: generate from A and B, judge
    scores register/voice, keep best above threshold; teacher-A rows must
    also pass the Decontaminator.
- `personaforge/eval/leakage.py` — leakage probes + scorer: passage-completion
  traps, protected-name probes, "continue this famous tale" traps; a reply
  fails on any blocklist hit or judge-recognized protected content.
  **Ship gate: 100% pass.**
- Provenance report writer: JSON (teacher mix, filter stats, leakage results)
  emitted by the build and eval runs.

**Private side (tolkien-llm / Spark)** — supplies at runtime:

- Teacher A: the tuned 27B served as `tolkien` on the llama-swap front door.
- Reference corpus for the overlap gate: `data/raw/tolkien.txt` (private).
- Blocklist: `data/blocklist.txt` (private; seeded manually + extracted
  capitalized-token frequency pass over the corpus, curated).
- PD CPT corpus: the existing `data/gutenberg/` set.

## Training

persona-forge CPT→SFT→DPO recipe, student **Qwen3.5-9B** (Apache-2.0;
q4 ≈ 5–6 GB — runs almost anywhere, which is the point). 27B variant later
if the 9B proves out. CPT on the PD gutenberg corpus; SFT+DPO on the filtered
blend (register chat + tales + mythic-pack roleplay/provocation/assignment,
plus DPO pairs from the existing generators with blocklist-safe content).
Volumes (env-tunable): ~800 register chat, ~400 tales, ~600 roleplay,
~400 DPO; teacher-A share smaller than teacher-B (q8 27B is ~10 tok/s;
gemma is fast) with the judge equalizing quality.

## Eval

- Existing persona-forge battery (held-out seeds): voice, in-character,
  assignment, boilerplate — same thresholds as the pack smoke.
- **Leakage eval: 100% required.** Runs pre-ship and its results go in the
  provenance report.
- Register check: voice_mean ≥ 0.75 against the judge.

## Ship (staged)

1. Merge 9B → GGUF q8 + MLX q4.
2. Beta: deliver to the trusted tester (his choice of artifact; over the
   existing tunnel or direct file).
3. Public: HF release under the persona-forge banner **only after user
   go-ahead post-beta** — model card + provenance report; MLflow registration
   in the private registry for lineage either way.
4. Naming: original coined name chosen at release time (user's call);
   nothing referencing the inspiring author.

## Out of scope

- HF publication without explicit user go-ahead after beta.
- 27B student (revisit after 9B).
- RAG/lore layer (the shareable model is voice-only by design).
- Sharing any private artifact: corpus, blocklist, tuned-teacher weights.
