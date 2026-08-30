"""
04_tts_voices_demo.py
=====================
Sarvam AI - Text-to-Speech Demo (Bulbul v3)
Covers:
  - Generating speech in multiple Indian languages
  - Sampling male and female speaker voices
  - Sample rate options (8000 / 16000 / 24000 Hz)
  - enable_preprocessing for numbers, abbreviations
  - Output audio formats (wav default; mp3 also shown)
  - Saving audio files to disk

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/overview
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
# Helper
# ---------------------------------------------------------------------------
def save_audio(audio_bytes, filename: str):
    with open(filename, "wb") as f:
        f.write(bytes(audio_bytes))
    print(f"    Saved -> {filename}")


# ---------------------------------------------------------------------------
# 1. Basic Hindi TTS
# ---------------------------------------------------------------------------
def demo_basic_tts():
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Hindi TTS (Bulbul v3, male voice)")
    print("=" * 60)

    text = "नमस्ते! आपका EMI भुगतान 15 जनवरी को देय है।"
    print(f"  Text: {text}")

    response = client.text_to_speech.convert(
        target_language_code="hi-IN",
        text=text,
        model="bulbul:v3",
        speaker="anand",            # professional Hindi male
        speech_sample_rate=16000,   # 8000 / 16000 / 24000
        enable_preprocessing=True,  # normalise numbers, abbrevs
    )
    save_audio(response.audios[0], "output_hindi_anand.wav")


# ---------------------------------------------------------------------------
# 2. Compare multiple voices for the same sentence
# ---------------------------------------------------------------------------
def demo_voice_comparison():
    print("\n" + "=" * 60)
    print("DEMO 2: Voice Comparison (same text, different speakers)")
    print("=" * 60)

    text = "नमस्ते, मैं आपकी कैसे मदद कर सकता हूँ?"
    print(f"  Text: {text}\n")

    voices = {
        "anand":  "Male   – professional",
        "rahul":  "Male   – casual",
        "priya":  "Female – professional",
        "simran": "Female – warm",
    }

    for speaker, description in voices.items():
        response = client.text_to_speech.convert(
            target_language_code="hi-IN",
            text=text,
            model="bulbul:v3",
            speaker=speaker,
            speech_sample_rate=16000,
        )
        filename = f"voice_comparison_{speaker}.wav"
        save_audio(response.audios[0], filename)
        print(f"  {speaker:8s} ({description})")


# ---------------------------------------------------------------------------
# 3. Multiple languages
# ---------------------------------------------------------------------------
def demo_multilingual_tts():
    print("\n" + "=" * 60)
    print("DEMO 3: TTS in Multiple Indian Languages")
    print("=" * 60)

    samples = [
        ("hi-IN", "anand",  "आपका खाता सफलतापूर्वक बनाया गया है।"),
        ("gu-IN", "varun",  "તમારું ખાતું સફળતાપૂર્વક બનાવવામાં આવ્યું છે."),
        ("mr-IN", "kabir",  "तुमचे खाते यशस्वीरित्या तयार केले गेले आहे."),
        ("bn-IN", "mani",   "আপনার অ্যাকাউন্ট সফলভাবে তৈরি হয়েছে।"),
        ("kn-IN", "gokul",  "ನಿಮ್ಮ ಖಾತೆಯನ್ನು ಯಶಸ್ವಿಯಾಗಿ ರಚಿಸಲಾಗಿದೆ."),
    ]

    for lang, speaker, text in samples:
        print(f"\n  [{lang}] speaker={speaker}")
        print(f"  Text: {text}")
        response = client.text_to_speech.convert(
            target_language_code=lang,
            text=text,
            model="bulbul:v3",
            speaker=speaker,
            speech_sample_rate=16000,
            enable_preprocessing=True,
        )
        filename = f"tts_{lang.replace('-', '_')}.wav"
        save_audio(response.audios[0], filename)


# ---------------------------------------------------------------------------
# 4. Sample rate options
# ---------------------------------------------------------------------------
def demo_sample_rates():
    print("\n" + "=" * 60)
    print("DEMO 4: Sample Rate Options (8kHz telephony / 24kHz high quality)")
    print("=" * 60)

    text = "नमस्ते! यह एक नमूना संदेश है।"
    print(f"  Text: {text}\n")

    for rate in [8000, 16000, 24000]:
        response = client.text_to_speech.convert(
            target_language_code="hi-IN",
            text=text,
            model="bulbul:v3",
            speaker="priya",
            speech_sample_rate=rate,
        )
        filename = f"sample_rate_{rate}hz.wav"
        save_audio(response.audios[0], filename)
        print(f"  {rate} Hz -> {filename}")

    print("\n  Use 8000 Hz for telephony/IVR, 24000 Hz for high-quality apps.")


# ---------------------------------------------------------------------------
# 5. Numbers and preprocessing
# ---------------------------------------------------------------------------
def demo_preprocessing():
    print("\n" + "=" * 60)
    print("DEMO 5: enable_preprocessing – Numbers, Dates, Abbreviations")
    print("=" * 60)

    text = "Your balance is Rs. 5,32,450. Due date: 15/01/2025."
    print(f"  Text: {text}\n")

    for preprocess in [False, True]:
        response = client.text_to_speech.convert(
            target_language_code="en-IN",
            text=text,
            model="bulbul:v3",
            speaker="aditya",
            speech_sample_rate=16000,
            enable_preprocessing=preprocess,
        )
        filename = f"preprocess_{'on' if preprocess else 'off'}.wav"
        save_audio(response.audios[0], filename)
        print(f"  preprocessing={'ON ' if preprocess else 'OFF'} -> {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Text-to-Speech Demo (Bulbul v3)")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/text-to-speech/overview")
    print()
    print("All output WAV files will be saved in the current directory.")

    demo_basic_tts()
    demo_voice_comparison()
    demo_multilingual_tts()
    demo_sample_rates()
    demo_preprocessing()

    print("\nDone. Play the .wav files to compare voices and languages.")
