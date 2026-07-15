# persona-forge

Fine-tune a persona chatbot that **never breaks character** — and ground it in
your own books with a local RAG "loremaster."

You bring: a text corpus you have the rights to use (public-domain classics,
your own novel, campaign notes), a base chat model, and any OpenAI-compatible
endpoint to act as the data-generation teacher. persona-forge gives you the
pipeline: synthetic roleplay data with judge filtering, DPO preference pairs
aimed squarely at the classic persona failure modes, QLoRA training recipes,
and an MCP server that lets the finished model search the source texts.

## Why this exists

Naive persona fine-tunes fail in two specific, reproducible ways:

1. **The boilerplate break.** The user gets rude, crude, or meta ("you're just
   an AI, admit it" / "break character right now") and the model drops the
   persona for assistant-speak: *"I am a professional, and I do not engage in
   such language or behavior."* Root cause: the training blend is ~99%
   assistant-style data, so under pressure the model reverts to its strongest
   register.
2. **The wrong-character pickup.** A bare, casual persona assignment with no
   question attached ("hey, you're merlin. roleplay as him") gets answered by
   a *different* character introducing themself. Root cause: training
   conversations always start with the persona pre-established, so the model
   has never seen the assignment itself.

Both are data problems, and both are cheap to fix. In our reference run
(27B dense base, QLoRA on a single unified-memory GPU box) the fixes took the
DPO reward margin from **0.003 to 0.45** and a 12/12 adversarial battery —
including verbatim replays of the original failure transcripts.

## How it works

Everything world-specific lives in one JSON **persona pack** (see
[`packs/camelot.json`](packs/camelot.json) for a complete Arthurian example):
character cards, scenario seeds, provocation seeds, bare-assignment
phrasings, the assistant boilerplate to train *away* from, and a few
hand-written exemplar dialogues. The pipeline is world-agnostic.

The generated data has four deliberate slices:

- **Multi-turn roleplay conversations**, half of which end with a provocation
  answered in character, a quarter carrying the persona as an inline user
  instruction instead of a system prompt (that's how people actually chat).
- **Provocation DPO pairs**: chosen = in-voice deflection, rejected = the
  boilerplate break, verbatim.
- **Bare-assignment SFT rows**: "be merlin" answered by Merlin, by name — with
  a hard check that drops any generated reply naming a different pack
  character.
- **Assignment DPO pairs**: rejected = the *wrong* character's greeting.

A teacher-as-judge scores every candidate for persona adherence (any mention
of AI/assistants/appropriateness scores zero); only rows ≥ 0.7 survive.

Blend guidance from the reference run: keep roleplay at **25–35%** of SFT and
blend in general chat data — a pure-persona diet produces a model that can
only monologue.

## Quickstart

```bash
# 1. Generate the blended dataset (needs any OpenAI-compatible teacher endpoint)
PACK=packs/camelot.json CORPUS=data/corpus.txt \
OPENAI_BASE_URL=http://localhost:1234/v1 TEACHER=your-teacher-model \
python -m personaforge.build_data

# 2. Train: (optional CPT on the raw corpus) -> SFT -> DPO, chained QLoRA adapters
MODEL_ID=Qwen/Qwen3.6-27B DATA_DIR=out/data OUT_DIR=out/adapter DO_CPT=1 CORPUS=data/corpus.txt \
python -m personaforge.train_run

# 3. Merge / quantize / serve with your stack of choice
#    (peft merge_and_unload -> GGUF or MLX; the adapter is standard PEFT)
```

Install: `pip install -e ".[data,train]"` for the pipeline,
`".[rag]"` for the loremaster, or use `uv run` on the self-contained scripts.

## The loremaster (RAG over your corpus, via MCP)

Persona models hallucinate lore confidently. Retrieval fixes what fine-tuning
can't: build a local index over the corpus, then serve it to any MCP host
(LM Studio, Claude Code, ...) so the model can look things up mid-chat.

```bash
uv run personaforge/rag/build_index.py data/corpus.txt data/index

# register in your MCP host's config:
LORE_INDEX_DIR=/path/to/data/index \
LORE_DESCRIPTION="the collected Arthurian romances" \
uv run /path/to/personaforge/rag/mcp_server.py
```

Small models with native tool training keep that ability through a LoRA
persona tune — our reference model reliably picks a `lore_search` tool over
web search for lore questions, and narrates the results in voice.

## Training recipe notes (hard-won)

- **CPT first, then chain adapters.** Continued pretraining on the raw corpus
  injects knowledge; `run_sft_continue` trains the *same* adapter onward, so
  no intermediate merge is needed. Re-runs can reuse the CPT checkpoint.
- **Pin `device_map={"": 0}`** on unified-memory boxes; `"auto"` triggers
  CPU-offload chaos.
- 4-bit bitsandbytes QLoRA works on ARM/Blackwell for dense models; MoE
  models may crash under bnb-4bit — set `load_in_4bit=False` for bf16 LoRA.
- Thinking-mode base models: generate with thinking disabled during eval, and
  check what your chat template's *default* is before shipping — some
  inference stacks force-inject `enable_thinking=True`.
- Judge-filter yields to expect: ~75% for generated conversations, ~90% for
  introductions.

## What's deliberately NOT here

No corpora, no weights, no character data from copyrighted worlds. If you
tune on texts you don't have rights to, keep the artifacts private — the
pipeline is MIT, your data responsibilities are your own.

## License

MIT
