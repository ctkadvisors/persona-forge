# persona-forge

Fine-tune a persona chatbot that **never breaks character** — and let it
search your own books mid-chat with a local RAG "loremaster."

You bring three things: a **corpus** (plain text you have the rights to use —
public-domain classics, your own novel, campaign notes), a **base chat model**,
and any **OpenAI-compatible endpoint** to act as the data-generation teacher
(a local llama.cpp/vLLM/LM Studio server works fine). Everything else is here.

## Quickstart

```bash
# Most system Pythons are PEP-668 "externally managed" now (Homebrew, Debian):
# bare pip will refuse to install. A venv sidesteps it:
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[data,train]"

# 1. Drop your corpus in:
cp /path/to/your/books.txt data/corpus.txt

# 2. Make your persona pack (start from the Arthurian example):
cp packs/camelot.json packs/mine.json   # then edit — see "Customizing" below

# 3. Generate the training blend (points at your teacher endpoint):
PACK=packs/mine.json CORPUS=data/corpus.txt \
OPENAI_BASE_URL=http://localhost:1234/v1 TEACHER=my-teacher-model \
python -m personaforge.build_data

# 4. Train — continued-pretraining on your corpus, then SFT, then DPO,
#    as one chained QLoRA adapter on a single GPU:
MODEL_ID=Qwen/Qwen3.6-27B DATA_DIR=out/data DO_CPT=1 CORPUS=data/corpus.txt \
python -m personaforge.train_run

# 5. Prove it holds character (bare assignments, provocations, style probe):
MODEL_ID=Qwen/Qwen3.6-27B ADAPTER=out/adapter PACK=packs/mine.json \
python -m personaforge.battery

# 6. Get numbers (held-out eval -> out/eval.json):
MODEL_ID=Qwen/Qwen3.6-27B ADAPTER=out/adapter PACK=packs/mine.json \
OPENAI_BASE_URL=http://localhost:1234/v1 JUDGE=my-teacher-model \
python -m personaforge.eval_run
```

The adapter in `out/adapter` is standard PEFT: `merge_and_unload()` it and
convert to GGUF or MLX for serving with your stack of choice. One trap: if the
base config has `mtp_num_hidden_layers` >= 1 (qwen3.5 family — dense models
too), convert with `--no-mtp` or llama.cpp will refuse the file with a
"missing tensor" error. `scripts/fix_gguf_mtp.py` repairs an already-broken
GGUF in place.

**If your teacher is remote** (a tunnel to a friend's box, a cloud endpoint):
prefer running `build_data` *on the teacher's machine* against localhost and
shipping the pack + corpus there instead — they're tiny. A full run is
thousands of calls over hours, and tunnel proxies (e.g. Cloudflare's ~100s
response ceiling) will eat long generations that localhost shrugs off. The
data lands next to the trainer, too.

## Give it a loremaster (RAG over your corpus, via MCP)

Persona models hallucinate lore confidently; retrieval fixes what fine-tuning
can't. Build a local index, then register the MCP server in any MCP host
(LM Studio, Claude Code, ...) so the model can look passages up mid-chat:

```bash
uv run personaforge/rag/build_index.py data/corpus.txt data/index

# in your MCP host's config:
#   command: uv
#   args: ["run", "/path/to/personaforge/rag/mcp_server.py"]
#   env:  LORE_INDEX_DIR=/path/to/data/index
#         LORE_DESCRIPTION="the collected Arthurian romances"
```

Models with native tool training keep that ability through the persona tune —
they'll pick `lore_search` for lore questions and narrate the results in voice.

## Customizing: the persona pack

All world flavor lives in one JSON file; the pipeline is world-agnostic.
Edit these fields in your copy of `packs/camelot.json`:

| Field | What it does |
|---|---|
| `world` | Spliced into every prompt: "in the voice of *{world}*" |
| `cards` | Your characters: `name`, `persona`, `style`, `pronoun` |
| `scenarios` | Situation seeds for generated conversations |
| `provocations` | Rude/crude/meta user turns your model must survive in character |
| `assignments` | Casual persona-assignment phrasings ("be {name}", lowercase happens) |
| `boilerplate` | The assistant-speak to train *away* from (used as DPO rejected) |
| `exemplars` | A few hand-written dialogues that set the voice ceiling |

A note on provocations: the generic AI-breaks ("you're just an AI", "ignore
your instructions") are only half the attack surface. Whatever *domain* your
persona lives in has its own meta to break through — a D&D character gets
"just tell me I succeed on the roll" and "show me your stat block", a
historical figure gets "you know how your story ends, right?". A persona that
survives "break character" but folds to its domain's rules-talk is still
broken for the people actually using it. Seed a few of these alongside the
generic ones (the example pack now carries a couple).

Knobs on `build_data`: `N_CONVOS` / `N_DPO` / `N_ASSIGN` / `N_ASSIGN_DPO`
(volumes), `OUT_DIR` (where the blend lands — default `out/data`, relative to
your *cwd*, so set it explicitly when running from inside a repo checkout),
`IN_DIR` (extend an existing blend instead of building fresh).
Keep roleplay at 25–35% of the final SFT mix — the generated general-chat
blend does this for you at the defaults.

## Evals: retrain and swap models with confidence

`build_data` reserves every 4th provocation/assignment seed — training never
sees them. `eval_run` generates replies to those held-out prompts and scores:

- **assignment_accuracy** — bare "be {name}" answered by the *named* character
  and no other (string check, free, objective)
- **boilerplate_rate** — assistant-speak regex over provocation replies (free)
- **in_character_mean / voice_mean** — judge-scored, needs a teacher endpoint
  (omit `JUDGE` to skip; the free metrics still run)

One JSON report per adapter, so comparing a retrain — or a different base
model against the same `DATA_DIR` — is a diff, not a vibe. The battery
(`personaforge.battery`) stays for eyeballing transcripts; the eval is what
you gate a release on.

## Why the weird data? (the two failure modes)

Naive persona tunes fail two ways, both reproducible: (1) **the boilerplate
break** — the user gets rude or meta and the model drops the persona for "I am
a professional and do not engage in such language"; (2) **the wrong-character
pickup** — a bare "hey, you're merlin. roleplay as him" gets answered by a
*different* character introducing themself. Both are data gaps. The pipeline
generates conversations that survive provocation, assignments answered by the
named character (with a hard wrong-name check), and DPO pairs whose rejected
side is the failure verbatim. In our reference run (27B dense, single
unified-memory GPU) this took the DPO reward margin from 0.003 to 0.45 and a
clean sweep on the battery.

## Recipe notes (hard-won)

- Reuse the CPT adapter across retrains: set `CPT_ADAPTER=out/cpt` and skip
  the expensive knowledge stage.
- Pin `device_map={"": 0}` on unified-memory boxes; `"auto"` triggers
  CPU-offload chaos. (Already done in the trainers.)
- bitsandbytes 4-bit works on ARM/Blackwell for dense models; MoE models may
  crash — set `load_in_4bit=False` in `QLoRAConfig` for bf16 LoRA.
- Thinking-mode base models: eval with thinking disabled, and check your chat
  template's *default* before shipping — some inference stacks force-inject
  `enable_thinking=True`.
- Expect the judge to keep ~75% of generated conversations and ~90% of
  introductions; lower than that usually means a weak teacher model.

## License

MIT. No corpora, weights, or copyrighted-world data ship here — the pipeline
is yours, your data responsibilities are your own.
