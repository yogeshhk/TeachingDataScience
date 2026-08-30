"""
02_stt_modes_demo.py
====================
Sarvam AI - Speech-to-Text Demo (Saaras v3)
Covers:
  - All 5 output modes: transcribe, translate, verbatim, translit, codemix
  - Auto language detection vs explicit language code
  - Looping over multiple modes on the same audio file
  - Supported formats note and 30-second REST limit

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Audio file: Provide a WAV/MP3 file under 30 seconds of Indian-language speech.
            The script accepts an optional command-line argument:
               python 02_stt_modes_demo.py hindi_audio.wav
            If no argument is given it looks for "sample_audio.wav" in the
            current directory.

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview
"""

import os
import sys
from pathlib import Path
from sarvamai import SarvamAI

# ---------------------------------------------------------------------------
# Initialise client
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("ERROR: SARVAM_API_KEY environment variable not set.")

client = SarvamAI(api_subscription_key=API_KEY)

# Audio file to use (command-line arg or default)
AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "sample_audio.wav"

if not Path(AUDIO_FILE).exists():
    sys.exit(
        f"ERROR: Audio file '{AUDIO_FILE}' not found.\n"
        "Provide a WAV/MP3 file under 30 seconds of Indian-language speech.\n"
        "Usage: python 02_stt_modes_demo.py <audio_file>"
    )


# ---------------------------------------------------------------------------
# 1. Basic transcription with explicit language code
# ---------------------------------------------------------------------------
def demo_basic_transcription():
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Transcription (mode=transcribe, hi-IN)")
    print("=" * 60)

    with open(AUDIO_FILE, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode="transcribe",
            language_code="hi-IN",   # explicit; remove for auto-detect
        )
    print(f"  Transcript : {response.transcript}")
    print(f"  Language   : {getattr(response, 'language_code', 'N/A')}")


# ---------------------------------------------------------------------------
# 2. All 5 output modes on the same audio
# ---------------------------------------------------------------------------
def demo_all_modes():
    print("\n" + "=" * 60)
    print("DEMO 2: All 5 Output Modes on the Same Audio")
    print("=" * 60)

    modes = ["transcribe", "translate", "verbatim", "translit", "codemix"]
    mode_descriptions = {
        "transcribe": "Original language text",
        "translate":  "Translated to English",
        "verbatim":   "Word-for-word, no normalisation",
        "translit":   "Romanised pronunciation",
        "codemix":    "Code-mixed script (Hinglish etc.)",
    }

    for mode in modes:
        with open(AUDIO_FILE, "rb") as f:
            response = client.speech_to_text.transcribe(
                file=f,
                model="saaras:v3",
                mode=mode,
                # language_code omitted -> auto-detect
            )
        desc = mode_descriptions[mode]
        print(f"  [{mode:11s}] {desc}")
        print(f"             -> {response.transcript}")
        print()


# ---------------------------------------------------------------------------
# 3. Auto language detection
# ---------------------------------------------------------------------------
def demo_auto_detect():
    print("\n" + "=" * 60)
    print("DEMO 3: Auto Language Detection (no language_code)")
    print("=" * 60)

    with open(AUDIO_FILE, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode="transcribe",
            # No language_code -> Saaras v3 auto-detects
        )
    print(f"  Transcript      : {response.transcript}")
    print(f"  Detected lang   : {getattr(response, 'language_code', 'check response object')}")


# ---------------------------------------------------------------------------
# 4. Translate mode: one-shot transcribe + translate to English
# ---------------------------------------------------------------------------
def demo_translate_mode():
    print("\n" + "=" * 60)
    print("DEMO 4: One-Shot Speech-to-English (mode=translate)")
    print("=" * 60)

    with open(AUDIO_FILE, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode="translate",
        )
    print(f"  English transcript : {response.transcript}")
    print("  (No separate translation API call needed)")


# ---------------------------------------------------------------------------
# Supported formats reminder
# ---------------------------------------------------------------------------
def print_format_notes():
    print("\n" + "=" * 60)
    print("NOTES")
    print("=" * 60)
    print("  Supported audio formats : wav, mp3, aac, aiff, ogg/opus,")
    print("                            flac, mp4/m4a, amr")
    print("  Max REST request length : 30 seconds")
    print("  For longer audio        : use Batch STT (see 06_call_analytics.py)")
    print("  WebSocket streaming     : see 03_stt_streaming.py")
    print("  Rate limit (Starter)    : 60 req/min")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI – Speech-to-Text Demo (Saaras v3)")
    print(f"Audio file : {AUDIO_FILE}")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview")

    demo_basic_transcription()
    demo_all_modes()
    demo_auto_detect()
    demo_translate_mode()
    print_format_notes()

    print("\nDone.")
