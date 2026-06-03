# Contributing

Thanks for wanting to add to the brain! The most useful contribution is usually
**new intents** — and you can write those without touching any Python.

## Adding an intent (no code)

Every intent lives in `brain_intents.json` under `"intents"`. Add an object and
it works on the next run. A minimal one:

```json
"coffee": {
  "keywords": ["coffee", "espresso", "latte"],
  "examples": ["I need a coffee", "make me a brew", "time for caffeine"],
  "responses": ["Starting a fresh pot ☕", "One coffee, coming up."]
}
```

### Field reference

| Field | Required | What it does |
|-------|----------|--------------|
| `keywords` | recommended | Core trigger words. **Synonym-expanded**, so list the obvious ones — `hour` already implies `time`. |
| `examples` | recommended | Whole phrasings a user might say. Matched by **meaning**, so 3–6 varied ones beat a long list of near-duplicates. |
| `responses` | yes | Reply templates; one is chosen at random. Use `{slot}` placeholders and `{name_suffix}` (expands to " <name>" if known). |
| `slots` | no | Live values to fill in: `current_time`, `current_date`, `math_eval`, `user_name`, `last_intent`. |
| `needs_number` | no | If `true`, the brain asks for a number before answering (good for math-style intents). |
| `requires` | no | Makes the intent a **composite** (see below). |

### Composite intents (combining concepts)

A composite fires **only when all listed concepts appear together**, and it
outranks single-concept intents because it's more specific:

```json
"plan_weather": {
  "requires": ["plan", "weather"],
  "responses": ["Want me to plan around the forecast? Tell me the day."]
}
```

`requires` values can be concept words (`plan`, `weather`) or context tags the
brain computes on its own: time-of-day (`@morning`, `@night`, `@late_night`),
day (`@friday`, `@weekend`), season (`@winter`…), holidays (`@christmas`,
`@halloween`…), and weather (`@rainy`, `@hot`…). Example: `"requires": ["hello", "@morning"]`
greets differently before noon.

## Adding synonyms

To teach the brain that two words mean the same thing, either edit
`BASE_SYNONYMS` in `brain.py`, or create an optional `brain_synonyms.json`:

```json
{ "coffee": ["coffee", "espresso", "latte", "brew", "java"] }
```

## Before you open a PR

1. Run `python demo.py` — it should still complete without errors.
2. Run `python brain.py` and try your new intent a few different ways
   (different wording, with a typo, combined with another intent).
3. Keep `responses` short (1–3 sentences) and in a neutral voice; character
   belongs in `brain_personas.json`, not the base intents.
4. Don't commit runtime files (`brain_memory.json`, `brain_learned.json`,
   `brain_unknowns.json`) — they're gitignored.

## Reporting bugs / ideas

Open an issue with what you typed, what the brain replied (intent + confidence
shown in brackets), and what you expected.
