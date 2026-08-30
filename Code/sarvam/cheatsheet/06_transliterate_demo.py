"""
06_transliterate_demo.py
========================
Sarvam AI - Transliteration and Language Identification Demo
Covers:
  - Indic -> Roman (romanisation / Romanisation)
  - Roman -> Indic (for users typing in Roman script)
  - spoken_form conversion (text to how it sounds when spoken)
  - numerals_format: international (0-9) vs native script numerals
  - spoken_form_numerals_language: English vs native number words
  - Language Identification: detect language AND script from raw text

Important limits:
  - Max 1000 characters per transliteration request
  - Transliteration between two Indic scripts (e.g. hi-IN -> gu-IN) is NOT supported
  - Only Indic <-> Roman (en-IN) is supported

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-processing/translation
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
# 1. Indic -> Roman (romanisation)
# ---------------------------------------------------------------------------
def demo_indic_to_roman():
    print("\n" + "=" * 60)
    print("DEMO 1: Indic Script -> Roman (Romanisation)")
    print("=" * 60)

    samples = [
        ("hi-IN", "नमस्ते, मेरा नाम विनायक है।"),
        ("gu-IN", "નમસ્તે, મારું નામ વિનાયક છે."),
        ("mr-IN", "नमस्कार, माझे नाव विनायक आहे."),
        ("bn-IN", "হ্যালো, আমার নাম বিনায়ক।"),
        ("kn-IN", "ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು ವಿನಾಯಕ."),
    ]

    for lang, text in samples:
        response = client.text.transliterate(
            input=text,
            source_language_code=lang,
            target_language_code="en-IN",   # always en-IN for Roman output
        )
        print(f"  [{lang}] {text}")
        print(f"        -> {response.transliterated_text}")
        print()


# ---------------------------------------------------------------------------
# 2. Roman -> Indic
# ---------------------------------------------------------------------------
def demo_roman_to_indic():
    print("\n" + "=" * 60)
    print("DEMO 2: Roman -> Indic Script (users typing in Roman)")
    print("=" * 60)

    samples = [
        ("hi-IN", "Main office ja raha hoon"),
        ("gu-IN", "Have tamara ghar kayn chhe?"),
        ("mr-IN", "Maza nav Vinayak ahe"),
    ]

    for lang, text in samples:
        response = client.text.transliterate(
            input=text,
            source_language_code="en-IN",
            target_language_code=lang,
        )
        print(f"  Roman ({lang}) : {text}")
        print(f"  Indic         : {response.transliterated_text}")
        print()


# ---------------------------------------------------------------------------
# 3. spoken_form: convert written form to how it sounds when spoken
# ---------------------------------------------------------------------------
def demo_spoken_form():
    print("\n" + "=" * 60)
    print("DEMO 3: spoken_form=True (written -> how it is spoken)")
    print("=" * 60)

    text = "मेरे पास 200 रुपये हैं।"
    print(f"  Source (hi-IN): {text}\n")

    for spoken in [False, True]:
        response = client.text.transliterate(
            input=text,
            source_language_code="hi-IN",
            target_language_code="en-IN",
            spoken_form=spoken,
        )
        label = "spoken_form=True " if spoken else "spoken_form=False"
        print(f"  [{label}] -> {response.transliterated_text}")


# ---------------------------------------------------------------------------
# 4. numerals_format: international (0-9) vs native script
# ---------------------------------------------------------------------------
def demo_numeral_formats():
    print("\n" + "=" * 60)
    print("DEMO 4: numerals_format (international vs native)")
    print("=" * 60)

    text = "मेरा phone number है 9840950950"
    print(f"  Source (hi-IN): {text}\n")

    for fmt in ["international", "native"]:
        response = client.text.transliterate(
            input=text,
            source_language_code="hi-IN",
            target_language_code="en-IN",
            numerals_format=fmt,
        )
        print(f"  [{fmt:13s}] -> {response.transliterated_text}")


# ---------------------------------------------------------------------------
# 5. spoken_form_numerals_language: English vs native number words
# ---------------------------------------------------------------------------
def demo_spoken_numeral_language():
    print("\n" + "=" * 60)
    print("DEMO 5: spoken_form_numerals_language (English vs native)")
    print("=" * 60)

    text = "मेरे पास 200 रुपये हैं।"
    print(f"  Source (hi-IN): {text}\n")

    for num_lang in ["english", "native"]:
        response = client.text.transliterate(
            input=text,
            source_language_code="hi-IN",
            target_language_code="en-IN",
            spoken_form=True,
            spoken_form_numerals_language=num_lang,
        )
        print(f"  [{num_lang:7s}] -> {response.transliterated_text}")

    print("\n  'english' -> 'do sau rupaye hain' (two hundred)")
    print("  'native'  -> 'do sau rupaye hain' (native words)")


# ---------------------------------------------------------------------------
# 6. Language Identification
# ---------------------------------------------------------------------------
def demo_language_identification():
    print("\n" + "=" * 60)
    print("DEMO 6: Language Identification (detect language + script)")
    print("=" * 60)

    samples = [
        "नमस्कार, तुम्ही कसे आहात?",   # Marathi (Devanagari)
        "નમસ્તે, તમે કેમ છો?",          # Gujarati (Gujarati script)
        "Hello, how are you?",          # English (Latin)
        "ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?",  # Kannada
        "Main theek hoon",              # Hindi in Roman script
    ]

    for text in samples:
        response = client.text.identify_language(input=text)
        lang    = getattr(response, "language_code", "N/A")
        script  = getattr(response, "script_code",   "N/A")
        print(f"  Input  : {text}")
        print(f"  Lang   : {lang}   Script : {script}")
        print()


# ---------------------------------------------------------------------------
# 7. Long text chunking (>1000 char limit)
# ---------------------------------------------------------------------------
def demo_long_text_chunking():
    print("\n" + "=" * 60)
    print("DEMO 7: Long Text Chunking (API limit = 1000 chars/request)")
    print("=" * 60)

    long_text = (
        "आर्टिफिशियल इंटेलिजेंस ने 21वीं सदी की सबसे परिवर्तनकारी तकनीकों में "
        "से एक के रूप में उभरा है। स्वास्थ्य सेवा से लेकर वित्त, परिवहन से लेकर "
        "शिक्षा तक, AI उद्योगों को फिर से आकार दे रहा है। " * 10  # make it >1000 chars
    )

    CHUNK_SIZE = 900   # stay safely under 1000-char limit

    chunks = [long_text[i:i + CHUNK_SIZE] for i in range(0, len(long_text), CHUNK_SIZE)]
    print(f"  Total chars : {len(long_text)}")
    print(f"  Chunks      : {len(chunks)} x {CHUNK_SIZE} chars max\n")

    results = []
    for idx, chunk in enumerate(chunks, 1):
        response = client.text.transliterate(
            input=chunk,
            source_language_code="hi-IN",
            target_language_code="en-IN",
        )
        results.append(response.transliterated_text)
        print(f"  Chunk {idx} -> {response.transliterated_text[:80]}...")

    full_result = " ".join(results)
    print(f"\n  Combined length: {len(full_result)} chars")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Transliteration and Language Identification Demo")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-processing/translation")
    print()
    print("NOTE: Transliteration between two Indic scripts is NOT supported.")
    print("      Only Indic <-> Roman (en-IN) is supported.")

    demo_indic_to_roman()
    demo_roman_to_indic()
    demo_spoken_form()
    demo_numeral_formats()
    demo_spoken_numeral_language()
    demo_language_identification()
    demo_long_text_chunking()

    print("\nDone.")
