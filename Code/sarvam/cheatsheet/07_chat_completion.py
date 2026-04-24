"""
07_chat_completion.py
=====================
Sarvam AI - Chat Completion Demo (Sarvam-M / Sarvam-30B / Sarvam-105B)
Covers:
  - Single-turn Q&A in multiple Indian languages
  - Multi-turn conversation with message history
  - System prompt customisation
  - Model comparison: sarvam-m (free) vs sarvam-30b vs sarvam-105b
  - Multilingual FAQ bot pattern (text in -> answer in same language)

Model pricing reminder:
  sarvam-m    -> FREE (no token charge); ideal for prototyping
  sarvam-30b  -> paid; balanced quality/cost
  sarvam-105b -> paid; flagship, highest quality

Rate limits (Starter plan):
  sarvam-m            : 60 req/min
  sarvam-30b/105b     : 40 req/min

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/chat-completion
"""

import os
import sys
from sarvamai import SarvamAI

# ---------------------------------------------------------------------------
# Initialise client
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("ERROR: SARVAM_API_KEY environment variable not set.")

client = SarvamAI(api_subscription_key=API_KEY)

FREE_MODEL    = "sarvam-m"       # always free
QUALITY_MODEL = "sarvam-30b"     # paid; swap to sarvam-105b for best quality


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def chat(messages: list, model: str = FREE_MODEL) -> str:
    response = client.chat.completions(messages=messages, model=model)
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# 1. Single-turn Q&A in multiple languages
# ---------------------------------------------------------------------------
def demo_single_turn_multilingual():
    print("\n" + "=" * 60)
    print("DEMO 1: Single-Turn Q&A in Multiple Languages (sarvam-m, free)")
    print("=" * 60)

    questions = [
        ("hi-IN", "Hindi",    "भारत की राजधानी कौन सी है?"),
        ("gu-IN", "Gujarati", "ભારતની રાજધાની કઈ છે?"),
        ("mr-IN", "Marathi",  "भारताची राजधानी कोणती आहे?"),
        ("bn-IN", "Bengali",  "ভারতের রাজধানী কী?"),
        ("en-IN", "English",  "What is the capital of India?"),
    ]

    for lang_code, lang_name, question in questions:
        reply = chat([
            {"role": "system",  "content": f"You are a helpful assistant. Reply only in {lang_name}."},
            {"role": "user",    "content": question},
        ])
        print(f"\n  [{lang_name}] Q: {question}")
        print(f"             A: {reply}")


# ---------------------------------------------------------------------------
# 2. Multi-turn conversation
# ---------------------------------------------------------------------------
def demo_multi_turn():
    print("\n" + "=" * 60)
    print("DEMO 2: Multi-Turn Conversation in Hindi (sarvam-m)")
    print("=" * 60)

    history = [
        {"role": "system", "content": "You are a helpful banking assistant. Reply in Hindi."},
    ]

    turns = [
        "मेरा बैंक बैलेंस कैसे चेक करूँ?",
        "नेट बैंकिंग में लॉगिन कैसे करते हैं?",
        "पासवर्ड भूल गया तो क्या करूँ?",
    ]

    for user_msg in turns:
        history.append({"role": "user", "content": user_msg})
        reply = chat(history)
        history.append({"role": "assistant", "content": reply})

        print(f"\n  User : {user_msg}")
        print(f"  Bot  : {reply}")

    print(f"\n  Total turns in history: {len(history) - 1}")  # exclude system


# ---------------------------------------------------------------------------
# 3. Multilingual FAQ bot pattern
# ---------------------------------------------------------------------------
def demo_faq_bot():
    print("\n" + "=" * 60)
    print("DEMO 3: Multilingual FAQ Bot (detect language, reply in same language)")
    print("=" * 60)

    # First detect the language, then answer in the same language
    questions = [
        "How do I reset my UPI PIN?",
        "मेरा UPI PIN कैसे reset करूँ?",
        "મારો UPI PIN કેવી રીતે reset કરવો?",
        "माझा UPI PIN कसा reset करायचा?",
    ]

    for question in questions:
        # Step 1: Detect language
        lang_resp = client.text.identify_language(input=question)
        lang_code = getattr(lang_resp, "language_code", "en-IN")

        lang_names = {
            "en-IN": "English", "hi-IN": "Hindi", "gu-IN": "Gujarati",
            "mr-IN": "Marathi", "bn-IN": "Bengali",
        }
        lang_name = lang_names.get(lang_code, lang_code)

        # Step 2: Answer in detected language
        reply = chat([
            {"role": "system",  "content": f"You are a helpful banking FAQ bot. Always reply in {lang_name}."},
            {"role": "user",    "content": question},
        ])

        print(f"\n  Q ({lang_code}): {question}")
        print(f"  A           : {reply}")


# ---------------------------------------------------------------------------
# 4. Model comparison: sarvam-m vs sarvam-30b
# ---------------------------------------------------------------------------
def demo_model_comparison():
    print("\n" + "=" * 60)
    print("DEMO 4: Model Comparison – sarvam-m (free) vs sarvam-30b (paid)")
    print("=" * 60)

    prompt = "Explain the Pradhan Mantri Jan Dhan Yojana in simple Hindi in 2 sentences."
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Reply in Hindi."},
        {"role": "user",   "content": prompt},
    ]

    print(f"  Prompt: {prompt}\n")

    for model in [FREE_MODEL, QUALITY_MODEL]:
        reply = chat(messages, model=model)
        print(f"  [{model}]")
        print(f"    {reply}")
        print()


# ---------------------------------------------------------------------------
# 5. Structured output: ask for JSON response
# ---------------------------------------------------------------------------
def demo_structured_output():
    print("\n" + "=" * 60)
    print("DEMO 5: Structured Output (JSON response in Indian language context)")
    print("=" * 60)

    reply = chat([
        {
            "role": "system",
            "content": (
                "You are a data extraction assistant. "
                "Always respond with valid JSON only. No other text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Extract the following fields from this text and return as JSON: "
                "name, amount, due_date.\n\n"
                "Text: 'राजेश कुमार का EMI भुगतान ₹5,000 है, जो 15 जनवरी 2025 को देय है।'"
            ),
        },
    ])
    print(f"  JSON output:\n  {reply}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Chat Completion Demo")
    print("Note: sarvam-m model is FREE; sarvam-30b/105b are paid.")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/chat-completion")

    demo_single_turn_multilingual()
    demo_multi_turn()
    demo_faq_bot()
    demo_model_comparison()
    demo_structured_output()

    print("\nDone.")
