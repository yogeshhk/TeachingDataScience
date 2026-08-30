"""
01_translate_demo.py
====================
Sarvam AI - Text Translation Demo
Covers:
  - Basic English <-> Indian language translation
  - All three register modes: formal, modern-colloquial, colloquial
  - Sarvam-Translate model for all 23 languages (beyond the 11 Mayura covers)
  - Auto language detection on the source side
  - Native vs international numeral formatting

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"
Docs:     https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-processing/translation
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


# ---------------------------------------------------------------------------
# 1. Basic translation: English -> Hindi
# ---------------------------------------------------------------------------
def demo_basic_translation():
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Translation (English -> Hindi)")
    print("=" * 60)

    response = client.text.translate(
        input="Welcome to the Sarvam AI workshop!",
        source_language_code="en-IN",
        target_language_code="hi-IN",
        speaker_gender="Female",
    )
    print(f"  English : Welcome to the Sarvam AI workshop!")
    print(f"  Hindi   : {response.translated_text}")


# ---------------------------------------------------------------------------
# 2. Auto-detect source language
# ---------------------------------------------------------------------------
def demo_auto_detect():
    print("\n" + "=" * 60)
    print("DEMO 2: Auto-Detect Source Language")
    print("=" * 60)

    samples = [
        ("नमस्ते, आप कैसे हैं?",   "hi-IN"),
        ("નમસ્તે, તમે કેમ છો?",    "gu-IN"),
        ("नमस्कार, तुम्ही कसे आहात?", "mr-IN"),
    ]

    for text, expected_lang in samples:
        response = client.text.translate(
            input=text,
            source_language_code="auto",   # let Sarvam detect
            target_language_code="en-IN",
        )
        print(f"  Input ({expected_lang}) : {text}")
        print(f"  English              : {response.translated_text}")
        print()


# ---------------------------------------------------------------------------
# 3. Register modes: formal / modern-colloquial / colloquial
# ---------------------------------------------------------------------------
def demo_register_modes():
    print("\n" + "=" * 60)
    print("DEMO 3: Translation Register Modes (EN -> HI)")
    print("=" * 60)

    text = "Your account has been blocked due to non-payment."
    print(f"  Source: {text}\n")

    for mode in ["formal", "modern-colloquial", "colloquial"]:
        response = client.text.translate(
            input=text,
            source_language_code="en-IN",
            target_language_code="hi-IN",
            mode=mode,
        )
        print(f"  [{mode:20s}] : {response.translated_text}")


# ---------------------------------------------------------------------------
# 4. Numeral formatting: international vs native
# ---------------------------------------------------------------------------
def demo_numeral_formats():
    print("\n" + "=" * 60)
    print("DEMO 4: Numeral Formatting (EN -> GU)")
    print("=" * 60)

    text = "Your loan EMI is Rs. 5000 due on the 15th."

    for fmt in ["international", "native"]:
        response = client.text.translate(
            input=text,
            source_language_code="en-IN",
            target_language_code="gu-IN",   # Gujarati
            numerals_format=fmt,
        )
        print(f"  [{fmt:13s}] : {response.translated_text}")


# ---------------------------------------------------------------------------
# 5. Multi-language broadcast: same sentence in many languages
# ---------------------------------------------------------------------------
def demo_broadcast():
    print("\n" + "=" * 60)
    print("DEMO 5: Broadcast Same Text to Multiple Languages")
    print("=" * 60)

    text = "The government scheme provides free healthcare to all citizens."
    targets = {
        "hi-IN": "Hindi",
        "gu-IN": "Gujarati",
        "mr-IN": "Marathi",
        "bn-IN": "Bengali",
        "kn-IN": "Kannada",
    }

    for code, name in targets.items():
        response = client.text.translate(
            input=text,
            source_language_code="en-IN",
            target_language_code=code,
            mode="formal",
        )
        print(f"  {name:10s} ({code}) : {response.translated_text}")


# ---------------------------------------------------------------------------
# 6. Sarvam-Translate: reach all 23 languages (beyond Mayura's 11)
# ---------------------------------------------------------------------------
def demo_sarvam_translate_extended():
    print("\n" + "=" * 60)
    print("DEMO 6: Sarvam-Translate – Extended 23-Language Coverage")
    print("=" * 60)

    text = "मेरा नाम विनायक है।"   # "My name is Vinayak"
    extended_targets = {
        "sat-IN":  "Santali",
        "mai-IN":  "Maithili",
        "doi-IN":  "Dogri",
        "brx-IN":  "Bodo",
        "mni-IN":  "Manipuri",
    }

    print(f"  Source (hi-IN): {text}\n")
    for code, name in extended_targets.items():
        response = client.text.translate(
            input=text,
            source_language_code="hi-IN",
            target_language_code=code,
            model="sarvam-translate:v1",   # required for 23-lang model
        )
        print(f"  {name:10s} ({code}) : {response.translated_text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI – Translation Demo")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-processing/translation")

    demo_basic_translation()
    demo_auto_detect()
    demo_register_modes()
    demo_numeral_formats()
    demo_broadcast()
    demo_sarvam_translate_extended()

    print("\nDone.")
