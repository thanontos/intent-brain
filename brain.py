"""
The brain: understand what a person MEANS, then answer.

This is a rewrite of the old keyword-counting brain. The old version could
only fire an intent when the user happened to type one of a hand-listed set of
trigger words. This version understands *meaning*, in layers, and degrades
gracefully so it always runs on a clean Python with zero installs:

  LAYER 1  Semantic similarity (best).  If `sentence-transformers` is installed,
           every intent's example phrasings are turned into meaning-vectors and
           the user's sentence is matched by cosine similarity -- so "got the
           hour?" matches the time intent even with no shared keyword.
           Auto-detected; no hard dependency.

  LAYER 2  Lexical understanding (always on, no installs).  Words are lemmatized
           (running/ran/runs -> run) and expanded through a synonym map
           (hour -> time), then scored with IDF weighting + typo tolerance.
           This gives real paraphrase coverage without a model.

On top of the matcher:
  * NEGATION   "I am not happy" is distinguished from "I am happy".
  * MULTI-INTENT  "what's the time and the date?" answers both.
  * CONTEXT    bare follow-ups ("and tomorrow?") inherit the last intent and
               its slots (date offset, last numbers) across turns.
  * GENERATIVE FALLBACK  a true unknown can be sent to a local Ollama model or
               an OpenAI-compatible endpoint if one is available; otherwise the
               brain asks you to teach it the answer and remembers it.

Run it:           python brain.py
Optional power-ups (the brain detects them automatically):
  pip install sentence-transformers     # turns on Layer 1 semantic matching
  run Ollama locally (ollama serve)     # turns on generative answers
  or set BRAIN_LLM_URL / BRAIN_LLM_KEY / BRAIN_LLM_MODEL for any OpenAI-style API
"""

import json
import math
import os
import random
import re
import sys
import datetime
import difflib
import urllib.request
from pathlib import Path

# Locate our files whether running as a .py script or a bundled PyInstaller .exe.
if getattr(sys, "frozen", False):              # running as a standalone executable
    APP_DIR = Path(sys.executable).parent      # writable: the folder next to the exe
    BUNDLE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))  # read-only bundled data
else:
    APP_DIR = BUNDLE_DIR = Path(__file__).parent

# intents: prefer a user-editable copy beside the app, else the bundled default
INTENTS_FILE = APP_DIR / "brain_intents.json"
if not INTENTS_FILE.exists():
    INTENTS_FILE = BUNDLE_DIR / "brain_intents.json"
# runtime state is always written next to the app (the bundle dir is temporary)
MEMORY_FILE = APP_DIR / "brain_memory.json"
LEARNED_FILE = APP_DIR / "brain_learned.json"
UNKNOWN_LOG = APP_DIR / "brain_unknowns.json"
SYNONYM_FILE = APP_DIR / "brain_synonyms.json"   # optional user-editable overrides
# personas: prefer a user-editable copy beside the app, else the bundled default
PERSONAS_FILE = APP_DIR / "brain_personas.json"
if not PERSONAS_FILE.exists():
    PERSONAS_FILE = BUNDLE_DIR / "brain_personas.json"

MIN_CONFIDENCE = 0.35    # below this, the brain admits it doesn't know
FUZZY_CUTOFF = 0.82      # how close a typo must be to count as the word
SEMANTIC_THRESHOLD = 0.42  # cosine cutoff when the embedding model is active
HISTORY_MAX = 20


# =============================================================================
# LAYER 1 -- optional semantic backend (sentence-transformers)
# =============================================================================

class SemanticBackend:
    """Wraps a sentence-embedding model if one is installed; otherwise inert."""

    def __init__(self):
        self.model = None
        self._cache = {}
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            # small, fast, ~90MB; good enough for short intents
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self.model = None  # stay in lexical mode

    @property
    def available(self):
        return self.model is not None

    def encode(self, text):
        if text in self._cache:
            return self._cache[text]
        vec = self.model.encode(text, normalize_embeddings=True)
        vec = [float(x) for x in vec]
        self._cache[text] = vec
        return vec

    @staticmethod
    def cosine(a, b):
        return sum(x * y for x, y in zip(a, b))  # both are already normalized

    def best_score(self, text, example_phrases):
        """Highest cosine similarity between `text` and any example phrasing."""
        if not example_phrases:
            return 0.0
        q = self.encode(text)
        return max(self.cosine(q, self.encode(p)) for p in example_phrases)


SEMANTIC = SemanticBackend()


# =============================================================================
# LAYER 2 -- lexical understanding: lemmatize + synonyms + IDF (no installs)
# =============================================================================

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "to",
    "of", "in", "on", "for", "and", "or", "what", "whats", "who", "how", "why",
    "when", "where", "me", "you", "your", "my", "i", "it", "this", "that",
    "about", "tell", "can", "could", "would", "please", "some", "more", "be",
}

# Concept groups: every word on a line is treated as the same meaning. This is
# what lets "hour", "clock" and "time" all light up the time intent offline.
# Users can extend this with brain_synonyms.json (same shape: {canonical: [...]})
BASE_SYNONYMS = {
    "time": ["time", "clock", "hour", "oclock", "moment"],
    "date": ["date", "day", "today", "calendar"],
    "plan": ["plan", "plans", "planned", "planning", "agenda", "schedule",
             "scheduled", "todo", "task", "tasks", "appointment", "errand",
             "errands", "itinerary", "checklist"],
    "hello": ["hello", "hi", "hey", "yo", "hiya", "howdy", "sup", "greetings",
              "morning", "evening", "afternoon"],
    "bye": ["bye", "goodbye", "later", "cya", "farewell", "goodnight", "night"],
    "thanks": ["thanks", "thank", "thx", "appreciate", "cheers", "ty", "grateful"],
    "add": ["add", "plus", "sum", "total", "combine"],
    "subtract": ["subtract", "minus", "less", "difference", "take"],
    "multiply": ["multiply", "times", "product", "multiplied"],
    "divide": ["divide", "divided", "over", "quotient", "split"],
    "calculate": ["calculate", "compute", "math", "equals", "evaluate", "work"],
    "weather": ["weather", "rain", "raining", "sunny", "temperature", "forecast",
                "cold", "hot", "snow", "windy", "humidity", "climate"],
    "help": ["help", "assist", "command", "option", "ability", "feature",
             "capable", "capability", "able"],
    "identity": ["who", "name", "yourself", "identity"],
    "mood": ["how", "feeling", "feel", "doing", "going", "ok", "okay", "alright"],
    "good": ["good", "great", "happy", "fine", "well", "awesome", "nice", "love"],
    "bad": ["bad", "sad", "terrible", "awful", "angry", "upset", "hate", "tired"],
}


def load_synonyms():
    groups = {k: list(v) for k, v in BASE_SYNONYMS.items()}
    if SYNONYM_FILE.exists():
        try:
            with open(SYNONYM_FILE, encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    groups.setdefault(k, [])
                    groups[k].extend(w for w in v if w not in groups[k])
        except (OSError, ValueError):
            pass
    # build word -> canonical lookup
    lookup = {}
    for canon, words in groups.items():
        for w in words:
            lookup[w] = canon
    return lookup


SYN = load_synonyms()


def normalize(text):
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def tokens(text):
    return [t for t in normalize(text).split() if t]


def lemmatize(word):
    """Tiny rule-based stemmer: collapse common English inflections."""
    w = word
    for suf, repl in (("ing", ""), ("edly", ""), ("ied", "y"), ("ies", "y"),
                      ("ed", ""), ("es", ""), ("s", ""), ("ly", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: len(w) - len(suf)] + repl
            break
    return w


def concept(word, use_syn=True):
    """Map a raw word to its meaning-bucket: synonym group, else its lemma.

    use_syn=False stays literal (lemma only). We index intent *keywords* with
    synonyms (curated, safe) but intent *examples* without (so an incidental
    'today' in a weather example doesn't masquerade as the date concept).
    """
    if use_syn and word in SYN:
        return SYN[word]
    lem = lemmatize(word)
    if use_syn and lem in SYN:
        return SYN[lem]
    return lem


def concepts_of(text, drop_stop=True, use_syn=True):
    cs = []
    for t in tokens(text):
        if len(t) < 2:          # drop junk single chars ("what's" -> "what","s")
            continue
        if drop_stop and t in STOPWORDS:
            continue
        cs.append(concept(t, use_syn))
    return cs


# =============================================================================
# learned intents (taught live by the user)
# =============================================================================

def load_learned():
    if LEARNED_FILE.exists():
        try:
            with open(LEARNED_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def save_learned(learned):
    try:
        with open(LEARNED_FILE, "w", encoding="utf-8") as f:
            json.dump(learned, f, indent=2)
    except OSError:
        pass


def keywords_from(text):
    kws = [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]
    return kws or tokens(text)


def teach_intent(question, answer):
    learned = load_learned()
    kws = keywords_from(question)
    if not kws:                      # nothing teachable (e.g. blank question)
        return None, []
    name = None
    for ln, intent in learned.items():
        if set(kws) & set(intent.get("keywords", [])):
            name = ln
            break
    if name is None:
        base = "learned_" + (kws[0] if kws else "topic")
        name = base
        i = 2
        while name in learned:
            name = f"{base}_{i}"
            i += 1
    entry = learned.get(name, {"keywords": [], "examples": [], "slots": {},
                               "responses": [], "learned": True})
    for kw in kws:
        if kw not in entry["keywords"]:
            entry["keywords"].append(kw)
    if question not in entry.setdefault("examples", []):
        entry["examples"].append(question)
    if answer not in entry["responses"]:
        entry["responses"].append(answer)
    learned[name] = entry
    save_learned(learned)
    return name, kws


def log_unknown(text):
    log = []
    if UNKNOWN_LOG.exists():
        try:
            with open(UNKNOWN_LOG, encoding="utf-8") as f:
                log = json.load(f)
        except (OSError, ValueError):
            log = []
    log.append({"text": text, "at": datetime.datetime.now().isoformat(timespec="seconds")})
    try:
        with open(UNKNOWN_LOG, "w", encoding="utf-8") as f:
            json.dump(log[-200:], f, indent=2)
    except OSError:
        pass


def load_intents():
    with open(INTENTS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    for name, intent in load_learned().items():
        data["intents"][name] = intent
    _index_intents(data)
    return data


def _index_intents(data):
    """Precompute concept sets + IDF weights once per load (cheap, big speedup)."""
    intents = data["intents"]
    fallback = data["_meta"]["fallback_intent"]
    # document frequency of each concept across intents (for IDF)
    df = {}
    for name, intent in intents.items():
        if name == fallback:
            continue
        bag = set()
        # keywords: curated -> safe to synonym-expand
        for kw in intent.get("keywords", []):
            bag.update(concepts_of(kw, drop_stop=True, use_syn=True))
        # examples: literal content lemmas only (no synonym cross-pollution)
        for ex in intent.get("examples", []):
            bag.update(c for c in concepts_of(ex, drop_stop=True, use_syn=False)
                       if len(c) > 2)
        intent["_concepts"] = bag
        for c in bag:
            df[c] = df.get(c, 0) + 1
    n = max(1, sum(1 for k in intents if k != fallback))
    data["_idf"] = {c: math.log((n + 1) / (freq + 0.5)) + 1.0 for c, freq in df.items()}


# =============================================================================
# persistent memory
# =============================================================================

PERSIST_KEYS = {"name", "persona", "facts", "chat", "mood"}


def load_memory():
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            pass
    return {}


def save_memory(memory):
    keep = {k: memory[k] for k in PERSIST_KEYS if k in memory}
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(keep, f, indent=2)
    except OSError:
        pass


# =============================================================================
# negation
# =============================================================================

NEGATORS = {"not", "no", "never", "dont", "don", "cant", "cannot", "wont",
            "isnt", "arent", "aint", "nope", "nah", "without", "nothing"}


def is_negated(text):
    """True if the sentence carries a negation cue."""
    toks = tokens(text)
    if any(t in NEGATORS for t in toks):
        return True
    if re.search(r"\bn't\b", text.lower()) or "n't" in text.lower():
        return True
    return False


def is_question(text):
    """Heuristic: a literal request for info (vs. a remark or exclamation).
    'what's the time?' is a request; 'look at the time, it's time to go' is not."""
    if "?" in text:
        return True
    return bool(re.search(
        r"\b(what|whats|when|which|how|do|does|got|gotta|have|tell|give|need|want)\b",
        normalize(text)))


# =============================================================================
# intent detection (Layer 1 if available, else Layer 2), per clause
# =============================================================================

def keyword_hit(kw, norm_text, toks):
    if re.search(r"\b" + re.escape(kw) + r"\b", norm_text):
        return 1.0
    if " " not in kw and difflib.get_close_matches(kw, toks, n=1, cutoff=FUZZY_CUTOFF):
        return 0.6
    return 0.0


def lexical_score(text, intent, idf):
    """Meaning overlap between the sentence and an intent, IDF-weighted, 0..~1."""
    user_cs = concepts_of(text)
    if not user_cs:
        user_cs = [concept(t) for t in tokens(text)]
    user_set = set(user_cs)
    intent_cs = intent.get("_concepts", set())
    if not intent_cs or not user_set:
        return 0.0
    overlap = user_set & intent_cs
    if not overlap:
        # last resort: typo-tolerant raw keyword match (old behaviour)
        norm, toks = normalize(text), tokens(text)
        raw = sum(keyword_hit(kw, norm, toks) for kw in intent.get("keywords", []))
        return min(raw, 1.0) * 0.5
    num = sum(idf.get(c, 1.0) for c in overlap)
    den = sum(idf.get(c, 1.0) for c in user_set) or 1.0
    coverage = num / den                       # how much of the user's meaning is explained
    recall = len(overlap) / len(intent_cs)     # how much of the intent was hit
    return min(1.0, 0.7 * coverage + 0.3 * recall + 0.15 * (len(overlap) - 1))


def detect_intent(text, data, memory=None):
    """Return (intent_name, confidence, method) for a single clause."""
    intents = data["intents"]
    fallback = data["_meta"]["fallback_intent"]
    idf = data.get("_idf", {})
    # spoken concepts PLUS the current temporal context ('@late_night', '@winter',
    # '@christmas'...) PLUS the known weather ('@rainy', '@hot'...), so composite
    # intents can combine words with *when* it is and *what it's like out*.
    user_concepts = set(concepts_of(text)) | time_context()["concepts"]
    wx = (memory or {}).get("weather")
    if wx and wx.get("tag"):
        user_concepts.add(wx["tag"])

    scores = {}
    semantic = SEMANTIC.available
    method = "semantic" if semantic else "lexical"
    for name, intent in intents.items():
        if name == fallback:
            continue
        # COMPOSITION ("mix blue + yellow = green"): a composite intent declares
        # the concepts it needs together via "requires". It fires only when ALL
        # of them co-occur, and -- being more specific -- outranks any single-
        # concept intent. An unsatisfied composite simply can't fire.
        req = intent.get("requires")
        if req is not None:
            if all(c in user_concepts for c in req):
                base = 0.85 + 0.04 * len(req)        # specificity bonus
                if semantic:
                    phrases = list(intent.get("examples", [])) + list(intent.get("keywords", []))
                    base = max(base, SEMANTIC.best_score(text, phrases) if phrases else 0.0)
                scores[name] = min(1.0, base)
            continue
        if semantic:
            phrases = list(intent.get("examples", [])) + list(intent.get("keywords", []))
            sem = SEMANTIC.best_score(text, phrases) if phrases else 0.0
            lex = lexical_score(text, intent, idf)
            scores[name] = max(sem, 0.55 * sem + 0.45 * lex)  # blend, semantic-led
        else:
            scores[name] = lexical_score(text, intent, idf)
    threshold = SEMANTIC_THRESHOLD if semantic else MIN_CONFIDENCE

    if not scores:
        return fallback, 0.0, method
    name = max(scores, key=scores.get)
    top = scores[name]
    if top < threshold:
        return fallback, top, method
    return name, top, method


# =============================================================================
# multi-intent: split compound input into clauses
# =============================================================================

def split_clauses(text):
    """Break 'time and the date?' into ['time', 'the date']. Conservative."""
    parts = re.split(r"\b(?:and then|and also|and|also|plus|then|;|,|\?|\.)\b|[?;,.]",
                     text, flags=re.IGNORECASE)
    parts = [p.strip() for p in parts if p and p.strip()]
    # only treat as multi-clause if at least two parts carry real content
    meaningful = [p for p in parts if [t for t in tokens(p) if t not in STOPWORDS]]
    if len(meaningful) >= 2:
        return meaningful
    return [text]


# =============================================================================
# entity extraction
# =============================================================================

def extract_numbers(text):
    return [float(n) if "." in n else int(n) for n in re.findall(r"-?\d+\.?\d*", text)]


def extract_name(text):
    t = text.lower()
    cand = None
    for pat in (r"call me\s+([a-z][a-z'-]+)",
                r"my name is\s+([a-z][a-z'-]+)",
                r"\bi am\s+([a-z][a-z'-]+)",
                r"\bi'?m\s+([a-z][a-z'-]+)"):
        m = re.search(pat, t)
        if m:
            cand = m.group(1)
            break
    if not cand:
        return None
    if cand in {"not", "fine", "good", "ok", "okay", "great", "happy", "sad",
                "tired", "here", "back", "sorry", "the", "again", "now", "just",
                "really", "still", "called", "sure"}:
        return None
    return cand.capitalize()


def extract_date_offset(text):
    t = text.lower()
    if "day after tomorrow" in t:
        return 2
    if "day before yesterday" in t:
        return -2
    if "tomorrow" in t:
        return 1
    if "yesterday" in t:
        return -1
    if "today" in t or "now" in t:
        return 0
    return None


def time_context(now=None):
    """A quick read of the *human* context around this moment -- time of day,
    weekday/weekend, season and nearby holiday -- so the brain can reason like a
    person ('2:39am isn't a normal hour to be awake'). Returns concept tags
    (namespaced with '@' for composition) plus a short summary for the model."""
    now = now or datetime.datetime.now()
    h = now.hour
    clock = now.strftime("%I:%M %p").lstrip("0")
    weekday = now.strftime("%A")
    weekend = now.weekday() >= 5
    m, d = now.month, now.day

    if h < 5:
        band, should_sleep = "the dead of night", True
    elif h < 7:
        band, should_sleep = "very early morning", True
    elif h < 12:
        band, should_sleep = "morning", False
    elif h < 14:
        band, should_sleep = "midday", False
    elif h < 18:
        band, should_sleep = "the afternoon", False
    elif h < 22:
        band, should_sleep = "the evening", False
    else:
        band, should_sleep = "late at night", True

    season = ("winter" if m in (12, 1, 2) else "spring" if m in (3, 4, 5)
              else "summer" if m in (6, 7, 8) else "autumn")

    holiday = {
        (1, 1): "New Year's Day", (2, 14): "Valentine's Day",
        (3, 17): "St. Patrick's Day", (7, 4): "Independence Day",
        (10, 31): "Halloween", (12, 24): "Christmas Eve",
        (12, 25): "Christmas Day", (12, 31): "New Year's Eve",
    }.get((m, d))
    if holiday is None:
        if m == 12 and d <= 25:
            holiday = "the run-up to Christmas"
        elif m == 10 and d >= 24:
            holiday = "Halloween season"

    # the cultural "feel" of each weekday -- how a person tends to experience it
    day_feel = {
        "Monday": "the dreaded start of the work week",
        "Tuesday": "still early in the work week, grinding onward",
        "Wednesday": "hump day -- the midpoint, and it's all downhill to the weekend from here",
        "Thursday": "almost there: the day before Friday",
        "Friday": "the eve of the weekend, when people stay up late and have fun knowing there's no work tomorrow",
        "Saturday": "the weekend -- sleeping in, staying up late, enjoying yourself",
        "Sunday": "a recovery and wind-down day, getting ready for the week to start again",
    }[weekday]
    # staying up late on a Fri/Sat night is normal fun, not cause for worry
    weekend_night = should_sleep and weekday in ("Friday", "Saturday")

    tags = {f"@{season}", f"@{weekday.lower()}", "@weekend" if weekend else "@weekday"}
    tags.add("@morning" if 5 <= h < 12 else "@afternoon" if 12 <= h < 18
             else "@evening" if 18 <= h < 22 else "@night")
    if should_sleep:
        tags.add("@late_night")
        if weekend_night:
            tags.add("@weekend_night")
    holiday_tag = {
        "Halloween": "@halloween", "Halloween season": "@halloween",
        "Christmas Eve": "@christmas", "Christmas Day": "@christmas",
        "the run-up to Christmas": "@christmas", "Independence Day": "@july4th",
        "Valentine's Day": "@valentines", "New Year's Day": "@newyear",
        "New Year's Eve": "@newyear", "St. Patrick's Day": "@stpatricks",
    }.get(holiday)
    if holiday_tag:
        tags.add(holiday_tag)

    summary = f"It is {clock} on {weekday}, in {season}"
    if holiday:
        summary += f", and it is {holiday}"
    summary += f". {weekday} is {day_feel}. This is {band}"
    if should_sleep and weekend_night:
        summary += " -- but it's a weekend night, so staying up to enjoy himself is perfectly fine"
    elif should_sleep:
        summary += " -- not a normal hour for your Master to be awake; he should rest soon"
    summary += "."

    return {"clock": clock, "weekday": weekday, "weekend": weekend,
            "season": season, "holiday": holiday, "band": band, "day_feel": day_feel,
            "should_sleep": should_sleep, "concepts": tags, "summary": summary}


# weather she's been told about -> colours her mood. Offline: learned from what
# the Master mentions ("it's pouring", "so hot") or a 'weather ...' command.
WEATHER_WORDS = {
    "stormy": (["storm", "stormy", "thunder", "thunderstorm", "lightning"], "@stormy"),
    "snowy":  (["snow", "snowing", "snowy", "blizzard", "sleet"], "@snowy"),
    "rainy":  (["rain", "raining", "rainy", "pouring", "drizzle", "drizzling", "downpour"], "@rainy"),
    "hot":    (["scorching", "boiling", "sweltering", "heatwave", "humid", "heat", "so hot", "too hot", "burning up"], "@hot"),
    "cold":   (["freezing", "chilly", "frosty", "frigid", "frost", "so cold", "too cold", "bitter cold"], "@cold"),
    "sunny":  (["sunny", "sunshine", "clear skies", "bright out"], "@sunny"),
    "cloudy": (["cloudy", "overcast", "grey out", "gray out", "gloomy", "dreary"], "@cloudy"),
    "windy":  (["windy", "gusty", "breezy"], "@windy"),
}

WEATHER_GUIDANCE = {
    "stormy": "There's a storm out -- it can feel unsettling. Be soothing, close and comforting.",
    "snowy":  "It's snowy and cold -- keep things cozy: warm tea, blankets, snuggling, a little wonder at the snow.",
    "rainy":  "It's rainy and grey -- the kind of day that feels gloomy and draining. Be extra warm, cheering and cuddly and try to light up your Master's mood.",
    "hot":    "It's uncomfortably hot -- your Master is likely sweaty, tired and a little irritable. Offer cool drinks, a fan, shade and gentle, patient care.",
    "cold":   "It's cold out -- offer warmth: blankets, a hot drink, snuggling close.",
    "sunny":  "It's bright and sunny -- be cheerful and upbeat, maybe suggest enjoying the lovely weather together.",
    "cloudy": "It's overcast and grey -- a mellow, low-energy day; be gentle, warm and a touch cheering.",
    "windy":  "It's windy out -- a bit wild; be cozy and grounding.",
}


def detect_weather(text):
    """Spot a weather mention in a message; return its state dict or None."""
    low = " " + normalize(text) + " "
    for cond, (words, tag) in WEATHER_WORDS.items():
        if any((" " + w + " ") in low for w in words):
            return {"cond": cond, "tag": tag, "guidance": WEATHER_GUIDANCE[cond]}
    return None


# =============================================================================
# slot resolvers
# =============================================================================

def resolve_slot(kind, text, memory):
    if kind == "current_time":
        return datetime.datetime.now().strftime("%I:%M %p").lstrip("0")
    if kind == "current_date":
        offset = extract_date_offset(text) or 0   # this message only; no carryover
        day = datetime.date.today() + datetime.timedelta(days=offset)
        return day.strftime("%A, %B %d, %Y").replace(" 0", " ")
    if kind == "math_eval":
        return safe_math(text, memory)
    if kind == "user_name":
        return memory.get("name", "friend")
    if kind == "last_intent":
        return memory.get("last_intent", "nothing yet")
    return ""


def safe_math(text, memory=None):
    words = (text.lower()
             .replace("multiplied by", "*").replace("divided by", "/")
             .replace("plus", "+").replace("add", "+").replace("sum", "+")
             .replace("minus", "-").replace("subtract", "-")
             .replace("times", "*").replace("multiply", "*")
             .replace("divide", "/").replace("over", "/"))
    expr = re.sub(r"[^0-9+\-*/.() ]", "", words).strip()
    # context: "and times 2" reuses the last result as the left operand
    if memory and memory.get("last_number") is not None and re.match(r"^[*/+\-]", expr):
        expr = str(memory["last_number"]) + expr
    # "add 10 and 5" -> "+ 10 5": one leading operator, numbers with no operator
    # between them -> distribute the operator ("10 + 5")
    nums = re.findall(r"-?\d+\.?\d*", expr)
    ops = re.findall(r"[+\-*/]", expr)
    if len(ops) == 1 and len(nums) >= 2 and re.match(r"^\s*[+\-*/]", expr):
        expr = ops[0].join(nums)
    if not re.search(r"\d", expr):
        return "(no numbers found)"
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        out = int(val) if isinstance(val, float) and val.is_integer() else val
        if memory is not None:
            memory["last_number"] = out
        return str(out)
    except Exception:
        return "(couldn't compute that)"


# =============================================================================
# context: bare follow-ups like "and tomorrow?"
# =============================================================================

CONTEXTUAL_INTENTS = {"date", "time", "math"}

# social "glue" intents whose canned replies break immersion mid-conversation;
# while she's chatting (her last reply came from the model) these are handed
# back to her instead of firing a flat templated line.
CHATTY_INTENTS = {"affirm", "deny", "thanks", "farewell", "greeting", "mood"}


def is_bare_followup(text):
    toks = tokens(text)
    if not toks or len(toks) > 4:
        return False
    cues = {"and", "what", "about", "how", "then", "also", "too"}
    has_cue = any(t in cues for t in toks)
    has_anchor = extract_date_offset(text) is not None or bool(extract_numbers(text))
    return has_cue or has_anchor


# =============================================================================
# generative fallback (optional, auto-detected)
# =============================================================================

def _http_json(url, payload, headers, timeout=20):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# =============================================================================
# personas -- a layered character on top of the brain (pure stdlib, no installs)
#   * 'system' shapes the generative model's voice when one is available
#   * 'prefixes'/'suffixes' flavor ordinary templated replies so the character
#     comes through even with no model running. Additive only -- the underlying
#     answer (numbers, dates, names) is never altered.
# =============================================================================

# Embedded fallback so the brain stays in character even if brain_personas.json
# is missing (e.g. a trimmed build). brain_personas.json overrides this.
DEFAULT_PERSONAS = {
    "_meta": {"default": "plain"},
    "personas": {
        "plain": {
            "greeting": "brain online. What do you need?",
            "system": "You are a concise, friendly assistant called '{bot_name}'. "
                      "Answer in 1-3 sentences.",
            "prefixes": [""], "suffixes": [""],
        },
        "maid": {
            "greeting": "Welcome home, Master. I've been waiting for you - how may I serve you today?",
            "system": "You are {bot_name}, a devoted and gentle maid. You address the "
                      "user as 'Master', speak warmly, politely and a little playfully, "
                      "and you are always eager to help. Keep answers to 1-3 sentences "
                      "and stay in character.",
            "prefixes": ["", "", "Of course, Master. ", "Right away, Master. ", "As you wish, Master. "],
            "suffixes": ["", "", "", " ♪", " I'm always happy to help, Master."],
        },
    },
}


def load_personas():
    """Personas from brain_personas.json, layered over the embedded defaults.
    Always returns a usable structure -- never raises."""
    personas = dict(DEFAULT_PERSONAS)
    try:
        if PERSONAS_FILE.exists():
            disk = json.loads(PERSONAS_FILE.read_text(encoding="utf-8"))
            merged = dict(DEFAULT_PERSONAS["personas"])
            merged.update(disk.get("personas", {}))
            personas = {
                "_meta": {**DEFAULT_PERSONAS["_meta"], **disk.get("_meta", {})},
                "personas": merged,
            }
    except Exception:
        pass  # malformed file -> fall back to embedded defaults
    return personas


PERSONAS = load_personas()


def get_persona(memory):
    """Return (name, persona_dict) for the active character."""
    default = PERSONAS["_meta"].get("default", "plain")
    name = (memory or {}).get("persona", default)
    persona = PERSONAS["personas"].get(name) or PERSONAS["personas"].get("plain", {})
    return name, persona


def persona_system_prompt(memory, bot_name="brain"):
    """The instruction handed to a generative model for the active persona."""
    _, p = get_persona(memory)
    base = p.get("system", "You are a concise, friendly assistant called '{bot_name}'. "
                           "Answer in 1-3 sentences.")
    return base.replace("{bot_name}", bot_name)


def apply_persona(reply, memory):
    """Sprinkle the active persona's flavor onto a finished reply. Additive only."""
    name, p = get_persona(memory)
    if name == "plain":
        return reply
    prefix = random.choice(p.get("prefixes") or [""])
    suffix = random.choice(p.get("suffixes") or [""])
    return f"{prefix}{reply}{suffix}"


def update_mood(text, memory):
    """Nudge her feelings based on the latest message, so her drives actually
    move: praise warms her (affection up, jealousy down); mentions of being
    away or with other people make her jealous; jealousy cools over time.
    Stored 0-10 and fed into her reply so her tone genuinely shifts."""
    mood = memory.setdefault("mood", {"affection": 5, "jealousy": 0})
    low = " " + normalize(text) + " "
    praise = any(w in low for w in (
        " good ", " great ", " love ", " cute ", " proud ", " thank ", " thanks ",
        " sweet ", " adorable ", " best ", " amazing ", " praise ", " perfect ",
        " happy ", " miss ", " missed "))
    away = any(w in low for w in (
        " friend ", " friends ", " work ", " busy ", " later ", " out ", " party ",
        " someone ", " tomorrow ", " leave ", " leaving ", " bye ", " gtg "))
    if praise:
        mood["affection"] = min(10, mood["affection"] + 1)
        mood["jealousy"] = max(0, mood["jealousy"] - 2)
    if away:
        mood["jealousy"] = min(10, mood["jealousy"] + 2)
    else:
        mood["jealousy"] = max(0, mood["jealousy"] - 1)   # cools off when you stay
    return mood


def generate_answer(text, memory, hint=None):
    """Try a real language model for open-ended questions. None if unavailable.

    `hint` lets a composite intent steer the reply (e.g. "the user is asking
    about today's plans") so the model improvises in character on the right topic.
    """
    name = memory.get("name")
    _, p = get_persona(memory)
    persona = persona_system_prompt(memory)
    if name:
        persona += f" The user's name is {name}."
    if hint:
        persona += f" {hint}"
    facts = memory.get("facts") or []
    if facts:
        persona += " Things you remember about your Master: " + "; ".join(facts) + "."
    mood = memory.get("mood") or {}
    if mood:
        def _lvl(n):
            return ("very high" if n >= 8 else "high" if n >= 6
                    else "moderate" if n >= 3 else "low")
        persona += (
            f" YOUR MOOD right now: affection toward Master is {_lvl(mood.get('affection', 5))}, "
            f"jealousy is {_lvl(mood.get('jealousy', 0))}. Let this colour your tone -- "
            f"warmer, happier and more affectionate when affection is high; poutier, "
            f"clingier and a little sulky when jealousy is high -- but never state the numbers.")
    ctx = time_context()
    persona += (f" CURRENT CONTEXT: {ctx['summary']} Take this into account "
                f"naturally -- gently fuss if it's very late, match the season or "
                f"holiday mood -- without reciting it back like a clock.")
    wx = memory.get("weather")
    if wx:
        persona += f" WEATHER: {wx['guidance']}"
    # a persona may override the model + creativity ("temperature"); 0.95 = playful
    temperature = float(p.get("temperature", 0.7))
    persona_model = (p.get("model") or "").strip()
    history = list(memory.get("chat", []))[-8:]   # recent turns -> real memory

    # 1) OpenAI-compatible endpoint via env vars
    url = os.environ.get("BRAIN_LLM_URL")
    if url:
        try:
            key = os.environ.get("BRAIN_LLM_KEY", "")
            model = persona_model or os.environ.get("BRAIN_LLM_MODEL", "gpt-4o-mini")
            messages = [{"role": "system", "content": persona}]
            messages += [{"role": h["role"], "content": h["content"]} for h in history]
            messages.append({"role": "user", "content": text})
            data = _http_json(
                url.rstrip("/") + "/chat/completions",
                {"model": model, "messages": messages, "temperature": temperature},
                {"Authorization": f"Bearer {key}"})
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            pass

    # 2) local Ollama -- replay the recent conversation so she remembers context
    try:
        model = persona_model or os.environ.get("BRAIN_OLLAMA_MODEL", "llama3.2")
        convo = ""
        for h in history:
            who = "User" if h["role"] == "user" else "Assistant"
            convo += f"{who}: {h['content']}\n"
        prompt = f"{persona}\n\n{convo}User: {text}\nAssistant:"
        data = _http_json(
            "http://localhost:11434/api/generate",
            {"model": model, "prompt": prompt, "stream": False,
             "options": {"temperature": temperature, "repeat_penalty": 1.15}},
            {}, timeout=30)
        out = (data.get("response") or "").strip()
        return out or None
    except Exception:
        return None


# =============================================================================
# reply assembly
# =============================================================================

def build_reply(text, data, memory=None):
    memory = memory if memory is not None else {}

    # blank or punctuation-only input: do nothing, never enter teach-me
    if not tokens(text):
        return _out("unknown", 0.0,
                    "I didn't catch that -- type something and I'll help.")

    # 0) waiting for the user to teach an answer?
    teach_q = memory.pop("teach_pending", None)
    if teach_q is not None:
        if normalize(text) in {"skip", "cancel", "never mind", "nevermind"}:
            return _out("unknown", 0.0, "Okay, skipped. Nothing learned.")
        name, kws = teach_intent(teach_q, text)
        if name is None:
            return _out("unknown", 0.0, "Okay, nothing to learn there.")
        data["intents"] = load_intents()["intents"]
        _index_intents(data)
        memory["last_intent"] = name
        return _out(name, 1.0, f"Got it. I'll remember that for: {', '.join(kws)}.",
                    taught=True)

    # 1) waiting on a slot answer? (multi-turn slot filling)
    pending = memory.pop("pending", None)
    if pending == "math" and (extract_numbers(text) or re.match(r"^\s*[*/+\-]", text)):
        return _finish(data, "math", text, memory, confidence=1.0)

    # 2) learn the user's name if offered
    found_name = extract_name(text)
    if found_name:
        first_time = memory.get("name") != found_name
        memory["name"] = found_name
        save_memory(memory)
        # only short-circuit on a plain introduction ("my name is Sam"); if the
        # message also carries another intent ("I'm Sam, what's the time?"), fall
        # through so that intent is still answered (the name is already saved).
        # 'identity' here is just the word "name" grazing the identity intent, so
        # it counts as "no other intent".
        other = detect_intent(text, data, memory)[0]
        if other in (data["_meta"]["fallback_intent"], "identity",
                     "name_query", "name_intro"):
            memory["last_intent"] = "name_intro"
            msg = (f"Nice to meet you, {found_name}!" if first_time
                   else f"Got it, {found_name}.")
            return _out("name_intro", 1.0, msg, method="rule")

    # 2b) recall: "what's my name?" / "who am i?" -> answer, don't re-introduce
    low = normalize(text)
    if not found_name and (re.search(r"\bmy name\b", low) or
                           re.search(r"\bwho am i\b", low)):
        if memory.get("name"):
            memory["last_intent"] = "name_query"
            return _out("name_query", 1.0,
                        f"You told me your name is {memory['name']}.", method="rule")
        return _out("name_query", 1.0,
                    "You haven't told me your name yet -- say 'my name is ...'.",
                    method="rule")

    fallback = data["_meta"]["fallback_intent"]

    # CONVERSATIONAL MODE: in a character persona (not 'plain') with a model
    # available, let her actually TALK. Route everyday and social input -- greetings,
    # goodbyes, "ok", remarks, questions to her -- to the maid with the running
    # history, instead of firing flat canned lines that break the story. Only real
    # tools stay templated: arithmetic, a literal time/date question, name capture/
    # recall. If no model answers, we fall through to the original templated logic
    # (so offline still works and the test suite is unchanged).
    if get_persona(memory)[0] != "plain":
        whole, wconf, _wm = detect_intent(text, data, memory)
        is_tool = (whole == "math" or whole in {"name_query", "name_intro"})
        if not is_tool:
            idef = data["intents"].get(whole, {})
            hint = idef.get("llm_hint") if idef.get("route") == "llm" else None
            if whole in {"time", "date"}:
                if is_question(text):            # a real request for the time/date
                    hint = ("Your Master is asking for the current time/date. Tell "
                            "him warmly and in character, using the EXACT time and "
                            "date from CURRENT CONTEXT above.")
                else:                            # just a remark mentioning it
                    hint = ("Your Master made a remark mentioning the time or date "
                            "rather than asking for it. React in character; if what "
                            "he means is unclear, sweetly ask him what he meant.")
            gen = generate_answer(text, memory, hint=hint)
            if gen:
                memory["last_intent"] = "generated"
                return _out(whole if whole != fallback else "chat",
                            round(wconf, 2), gen, method="generative")

    # 3) multi-intent: only if clauses yield >=2 DISTINCT confident intents
    clauses = split_clauses(text)
    ordered = []
    if len(clauses) >= 2:
        seen = set()
        for clause in clauses:
            name, conf, method = detect_intent(clause, data, memory)
            if name != fallback and name not in seen:
                seen.add(name)
                ordered.append((clause, name, conf, method))
    if len(ordered) >= 2:
        replies, intents_hit = [], []
        for clause, name, conf, method in ordered[:3]:
            sub = _finish(data, name, clause, memory, conf, record=False)
            replies.append(sub["reply"])
            intents_hit.append(name)
        memory["last_intent"] = intents_hit[-1]
        return _out("+".join(intents_hit), round(max(c for _, _, c, _ in ordered), 2),
                    " ".join(replies), method=ordered[0][3])

    # 4) single intent -> detect on the WHOLE sentence (keeps math expressions
    #    and other phrasing intact rather than judging a fragment)
    name, conf, method = detect_intent(text, data, memory)
    if name == "identity" and not re.search(
            r"\b(you|your|yourself|u)\b", normalize(text)):
        name = fallback  # open-ended ("tell me about life") -> let fallback handle

    # incidental-keyword guard: a content-rich sentence that only grazes a
    # factual utility intent (date/time/weather) through a single word -- e.g.
    # "is there anything planned today" hitting 'date' via "today" -- is really
    # an open-ended question, so hand it to the generative model / fallback.
    if name in {"date", "time", "weather"}:
        content = [t for t in tokens(text) if t not in STOPWORDS and len(t) > 2]
        intent_cs = data["intents"][name].get("_concepts", set())
        explained = sum(1 for w in content if concept(w) in intent_cs)
        if len(content) >= 3 and explained / len(content) < 0.5:
            name = fallback

    if name != fallback:
        intent = data["intents"][name]
        # composite intent that routes to the maid LLM for an in-character reply
        if intent.get("route") == "llm":
            gen = generate_answer(text, memory, hint=intent.get("llm_hint"))
            if gen:
                memory["last_intent"] = name
                return _out(name, round(conf, 2), gen, method="generative")
            if intent.get("responses"):          # graceful offline fallback line
                return _finish(data, name, text, memory, conf, method="composite")
            log_unknown(text)
            memory["teach_pending"] = text
            memory["last_intent"] = fallback
            return _out("unknown", 0.0,
                        "I don't know that one yet. What should I say when you ask "
                        "that? (type the answer, or 'skip')", awaiting=True)
        # pragmatic ambiguity: a remark like "look at the time, it's time to go"
        # literally matches time/date but isn't a literal request. If she can talk
        # (model available), let her react / ask what Master actually meant rather
        # than reciting the clock. Offline, fall through to the literal answer.
        if name in {"time", "date"} and not is_question(text):
            gen = generate_answer(text, memory, hint=(
                "Your Master made a remark that mentions the time or date rather "
                "than asking you to tell it. Do NOT just recite the clock or date. "
                "React in character, and if what he means is unclear (is he leaving? "
                "tired? teasing? being serious or sarcastic?), sweetly ask him to "
                "clarify what he meant before going along with it."))
            if gen:
                memory["last_intent"] = "generated"
                return _out(name, round(conf, 2), gen, method="generative")
        if name == "name_query" and not memory.get("name"):
            return _out("name_query", round(conf, 2),
                        "You haven't told me your name yet -- say 'my name is ...'.",
                        method=method)
        # negation turns an affirmation into a denial ("no", "not right")
        if is_negated(text) and name == "affirm":
            name, intent = "deny", data["intents"].get("deny", intent)
        # missing required number -> ask for it
        if intent.get("needs_number") and not extract_numbers(text):
            memory["pending"] = "math"
            memory["last_intent"] = name
            return _out(name, round(conf, 2),
                        "Sure, which numbers? (e.g. 6 times 7)", awaiting=True,
                        method=method)
        return _finish(data, name, text, memory, conf, method=method)

    # 4a) bare follow-up inherits the last contextual intent + slots
    last = memory.get("last_intent")
    if last in CONTEXTUAL_INTENTS and is_bare_followup(text):
        return _finish(data, last, text, memory, confidence=0.7)

    # 5) unknown -> try a real model, else log + offer to learn
    gen = generate_answer(text, memory)
    if gen:
        memory["last_intent"] = "generated"
        return _out("generated", 0.5, gen, method="generative")

    log_unknown(text)
    memory["teach_pending"] = text
    memory["last_intent"] = fallback
    return _out("unknown", 0.0,
                "I don't know that one yet. What should I say when you ask that? "
                "(type the answer, or 'skip')", awaiting=True)


def _out(intent, confidence, reply, awaiting=False, method="lexical", **extra):
    d = {"intent": intent, "confidence": round(confidence, 2), "reply": reply,
         "awaiting": awaiting, "method": method}
    d.update(extra)
    return d


def _finish(data, intent_name, text, memory, confidence, record=True, method="lexical"):
    intent = data["intents"][intent_name]
    fills = {}
    for slot_key, kind in intent.get("slots", {}).items():
        fills[slot_key] = resolve_slot(kind, text, memory)
    fills["name_suffix"] = f" {memory['name']}" if memory.get("name") else ""

    if intent_name == "math" and str(fills.get("result", "")).startswith("("):
        reply = "I couldn't work that out -- try something like '6 times 7'."
        if record:
            memory["last_intent"] = intent_name
        return _out(intent_name, confidence, reply, method=method)

    template = random.choice(intent["responses"])
    try:
        reply = template.format(**fills)
    except (KeyError, IndexError):
        reply = template

    if record:
        memory["last_intent"] = intent_name
        hist = memory.setdefault("history", [])
        hist.append({"text": text, "intent": intent_name})
        del hist[:-HISTORY_MAX]
    return _out(intent_name, confidence, reply, method=method)


# =============================================================================
# REPL
# =============================================================================

def main():
    try:                       # let cute unicode (♡, ~, ehe) print on any console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    data = load_intents()
    memory = load_memory()
    mode = "semantic+lexical" if SEMANTIC.available else "lexical"
    back = []
    if SEMANTIC.available:
        back.append("embeddings ON")
    if os.environ.get("BRAIN_LLM_URL") or _ollama_up():
        back.append("generative ON")
    extra = (" [" + ", ".join(back) + "]") if back else \
            " (tip: install sentence-transformers or run Ollama for more power)"
    name = memory.get("name")
    pname, p = get_persona(memory)
    greeting = p.get("greeting", f"brain online ({mode}).")
    greeting = greeting.replace("{name}", name or "").replace("  ", " ").strip()
    print(greeting)
    print(f"[{mode} | persona: {pname}]{extra}")
    print("commands: 'persona <name>' (switch character; 'persona list' to see all), "
          "'quit', 'forget' (wipe your name), 'unlearn' (wipe taught topics), "
          "'learned' (list taught topics).\n")
    while True:
        try:
            text = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        cmd = text.lower()
        if cmd in {"quit", "exit"}:
            break
        if cmd == "forget":
            memory.clear()
            save_memory(memory)
            print("brain > Personal memory wiped (taught topics kept).\n")
            continue
        if cmd == "unlearn":
            save_learned({})
            data = load_intents()
            print("brain > Forgot everything you taught me.\n")
            continue
        if cmd == "learned":
            learned = load_learned()
            if not learned:
                print("brain > I haven't been taught anything yet.\n")
            else:
                for n, it in learned.items():
                    print(f"        - {', '.join(it['keywords'])} -> {it['responses'][0]!r}")
                print()
            continue
        if cmd == "persona" or cmd.startswith("persona "):
            parts = text.split()
            if len(parts) == 1 or parts[1].lower() in {"list", "?"}:
                names = ", ".join(PERSONAS["personas"].keys())
                cur = memory.get("persona", PERSONAS["_meta"].get("default", "plain"))
                print(f"brain > Personas available: {names}. Current: {cur}. "
                      f"Switch with 'persona <name>'.\n")
                continue
            choice = parts[1].lower()
            if choice in PERSONAS["personas"]:
                memory["persona"] = choice
                save_memory(memory)
                _, np_ = get_persona(memory)
                g = np_.get("greeting", "At your service.")
                g = g.replace("{name}", memory.get("name", "")).replace("  ", " ").strip()
                print(f"brain > {g}\n")
            else:
                print(f"brain > I don't have a '{choice}' persona yet. Try 'persona list'.\n")
            continue
        if cmd.startswith("remember "):
            fact = text[len("remember "):].strip()
            if fact:
                facts = memory.setdefault("facts", [])
                if fact not in facts:
                    facts.append(fact)
                save_memory(memory)
                print(f"brain > {apply_persona('Mm~ I will keep that close, Master.', memory)}\n")
            continue
        if cmd in {"facts", "remember"}:
            facts = memory.get("facts", [])
            if facts:
                print("brain > Things I remember about you:")
                for f in facts:
                    print(f"        - {f}")
                print()
            else:
                print("brain > I don't have anything saved about you yet -- tell me with 'remember ...'.\n")
            continue
        if cmd == "mood":
            m = memory.get("mood", {"affection": 5, "jealousy": 0})
            print(f"brain > [affection {m.get('affection', 5)}/10 | "
                  f"jealousy {m.get('jealousy', 0)}/10]\n")
            continue
        if cmd == "weather":
            wx = memory.get("weather")
            if wx:
                print(f"brain > Right now I'm treating the weather as {wx['cond']}.\n")
            else:
                print("brain > I don't know the weather yet -- just mention it "
                      "(e.g. \"it's pouring outside\") or type 'weather rainy'.\n")
            continue
        if cmd.startswith("weather "):
            wx = detect_weather(text[len("weather "):])
            if wx:
                memory["weather"] = wx
                print(f"brain > Mm, I'll keep in mind it's {wx['cond']} out, Master.\n")
            else:
                print("brain > I didn't catch a weather word there -- try rain, hot, "
                      "cold, snow, sunny, cloudy, windy...\n")
            continue
        if not text:
            continue
        update_mood(text, memory)
        wx_now = detect_weather(text)
        if wx_now:
            memory["weather"] = wx_now   # she remembers the day's weather
        out = build_reply(text, data, memory)
        # generative replies are already in-character; only flavor templated ones
        reply = out["reply"] if out.get("method") == "generative" \
            else apply_persona(out["reply"], memory)
        # remember the exchange so she has real conversational memory next turn
        chat = memory.setdefault("chat", [])
        chat.extend([{"role": "user", "content": text},
                     {"role": "assistant", "content": reply}])
        del chat[:-12]   # keep the last ~6 exchanges
        save_memory(memory)   # persist her memory of you for next time
        print(f"brain > {reply}   "
              f"[intent: {out['intent']} | conf: {out['confidence']} | {out['method']}]\n")


def _ollama_up():
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
