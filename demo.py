"""
demo.py -- a guided tour of what the brain does.

Run it:  python demo.py

No installs, no model, no internet needed. It feeds the brain a series of
example sentences and prints, for each one, the intent it detected and the
answer it built -- so you can watch it (1) understand meaning even when the
wording changes, (2) combine several intents in one sentence, and (3) fire a
COMPOSITE intent only when two concepts appear together.

When you're done, run `python brain.py` to chat with it yourself.
"""

import brain

# Use the neutral 'plain' persona so the demo output is clean and deterministic.
DATA = brain.load_intents()
MEMORY = {"persona": "plain"}


def ask(sentence):
    """Send one sentence to the brain and print what it understood."""
    out = brain.build_reply(sentence, DATA, dict(MEMORY))  # fresh memory each line
    intent = out["intent"]
    conf = out["confidence"]
    method = out["method"]
    print(f"  you  > {sentence}")
    print(f"  brain> {out['reply']}")
    print(f"         └─ intent: {intent!r:22}  confidence: {conf:<5}  via: {method}")
    print()


def section(title, blurb):
    print("=" * 70)
    print(title)
    print("-" * 70)
    print(blurb)
    print()


def main():
    print()
    print("  THE BRAIN — a tiny engine that understands what you MEAN.")
    print()

    section(
        "1. Same meaning, different words",
        "Each pair below uses NO shared keyword with the next, yet lands on the\n"
        "same intent. The brain matches meaning, not exact words.")
    ask("what time is it")
    ask("got the hour?")
    ask("what's the date today")
    ask("which day are we on")

    section(
        "2. It does the work, not just the matching",
        "Some intents fill in a live answer (the clock, the calendar, a sum).")
    ask("what's 15 times 8")
    ask("add 240 and 17")

    section(
        "3. Combining intents — many answers in one sentence",
        "Ask for several things at once and the brain splits the sentence into\n"
        "clauses, answers each, and stitches the replies together.")
    ask("tell me the time and the date")
    ask("hi there, what's the date, and what is 9 times 9?")

    section(
        "4. Combining concepts — a COMPOSITE intent",
        "'plan' alone hits the plan intent. 'weather' alone hits the weather\n"
        "intent. Mention BOTH and a more specific composite intent outranks them.\n"
        "This is the brain combining two concepts into one purpose-built answer.")
    ask("what's on my agenda")
    ask("what's the weather like")
    ask("should I plan my day around the weather")

    section(
        "5. Something it doesn't know",
        "Unknown input isn't faked. In the real prompt it offers to LEARN the\n"
        "answer from you and remembers it; here it simply admits it.")
    ask("what's the capital of Mars")

    print("=" * 70)
    print("Your turn:  python brain.py")
    print("Teach it new intents by editing brain_intents.json — no code required.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
