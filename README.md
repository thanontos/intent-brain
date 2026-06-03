# 🧠 The Brain

A tiny intent engine that understands what a person **means**, then answers — and **combines** intents to build one reply. Pure Python, **zero dependencies**, runs on a clean install. Optional power-ups (semantic embeddings, a local/remote LLM) are auto-detected if you have them, but nothing is required.

You bring the intents (a JSON file). The brain handles the understanding.

```
git clone https://github.com/classicbdo50/intent-brain.git
cd intent-brain
python demo.py     # watch it work
python brain.py    # chat with it yourself
```

## See it in 30 seconds

```
$ python demo.py
```

```
you  > got the hour?
brain> The time is 6:47 PM.
       └─ intent: 'time'         confidence: 1.0   via: lexical

you  > tell me the time and the date
brain> It's 6:47 PM. Today is Wednesday, June 3, 2026.
       └─ intent: 'time+date'    confidence: 0.8   via: lexical

you  > should I plan my day around the weather
brain> Planning around the weather, are we? ...
       └─ intent: 'plan_weather' confidence: 0.93  via: lexical
```

Note the first line: `got the hour?` shares **no keyword** with the intent's name yet still lands on `time`. That's the point.

## How it understands meaning

The brain works in layers and degrades gracefully — it always runs, and gets smarter if you give it more.

1. **Semantic similarity** *(optional, best)* — if `sentence-transformers` is installed, every intent's example phrasings become meaning-vectors and your sentence is matched by cosine similarity. `pip install sentence-transformers` to turn it on.
2. **Lexical understanding** *(always on, no installs)* — words are lemmatized (`running/ran/runs → run`), expanded through a synonym map (`hour → time`), scored with IDF weighting, and matched with typo tolerance. Real paraphrase coverage with no model.

On top of the matcher it also handles **negation** (`I am not happy` ≠ `I am happy`), **context** (bare follow-ups like `and tomorrow?` inherit the last intent), and an optional **generative fallback** to a local [Ollama](https://ollama.com) model or any OpenAI-compatible endpoint for true unknowns — otherwise it asks *you* to teach it the answer and remembers it.

## How it combines intents

This is the headline feature. There are two ways the brain merges intents into a single answer:

**1. Multi-intent — several requests in one sentence.** The brain splits a sentence into clauses, answers each, and stitches the replies together.

> `hi there, what's the date, and what is 9 times 9?`
> → `Hey there. Today is Wednesday, June 3, 2026. That comes to 81.`

**2. Composition — a `requires` intent.** A *composite* intent declares the concepts it needs together. It fires **only when all of them co-occur**, and — being more specific — outranks any single-concept intent. In `brain_intents.json`:

```json
"plan_weather": {
  "requires": ["plan", "weather"],
  "responses": ["Planning around the weather, are we? ..."]
}
```

`plan` alone hits the `plan` intent; `weather` alone hits `weather`; mention **both** and `plan_weather` wins. `requires` can also use context tags like `@morning`, `@winter`, or `@rainy`, so an intent can combine a word with *when* it is or *what it's like out*.

## Teach it your own intents — no code

Everything the brain knows lives in **`brain_intents.json`**. Add an entry and it works on the next run:

```json
"coffee": {
  "keywords": ["coffee", "espresso", "latte"],
  "examples": ["I need a coffee", "make me a brew"],
  "responses": ["Starting a fresh pot ☕", "On it — one coffee coming up."]
}
```

Field reference:

| Field | What it does |
|-------|--------------|
| `keywords` | Core words. **Synonym-expanded** — safe, curated. |
| `examples` | Natural phrasings, matched by **meaning**. |
| `responses` | One is picked at random. Supports `{slot}` placeholders. |
| `slots` | Live values: `current_time`, `current_date`, `math_eval`, `user_name`, `last_intent`. |
| `needs_number` | If `true`, the brain asks for a number before answering. |
| `requires` | Makes the intent a **composite** — fires only when all listed concepts co-occur. |

You can also extend the synonym map with an optional `brain_synonyms.json` (`{ "canonical": ["word", "word"] }`).

## Personas (optional flavor)

A character layer on top of the brain, in `brain_personas.json`. Ships neutral (`plain`); switch at runtime with `persona <name>` (a `butler` and a `maid` are included as examples). Personas only *add* flavor — numbers, dates, and names are never altered.

## Files

| File | Role |
|------|------|
| `brain.py` | The engine + interactive prompt. |
| `demo.py` | Guided, no-install tour of the features. |
| `brain_intents.json` | **What the brain knows. Edit this.** |
| `brain_personas.json` | Optional character layer. |
| `brain_synonyms.json` | Optional extra synonyms (not included; create if you want). |
| `brain_memory.json`, `brain_learned.json`, `brain_unknowns.json` | Auto-created at runtime; gitignored. |

## Optional power-ups

```bash
pip install sentence-transformers          # Layer 1 semantic matching
ollama serve                               # local generative answers
# or point at any OpenAI-compatible API:
export BRAIN_LLM_URL=...  BRAIN_LLM_KEY=...  BRAIN_LLM_MODEL=...
```

## License

MIT — see `LICENSE`.
