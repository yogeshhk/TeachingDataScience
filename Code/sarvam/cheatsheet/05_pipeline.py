"""
05_pipeline.py
==============
Sarvam AI - Full End-to-End Voice Pipeline
  Speech In  ->  Saaras v3 STT  ->  Sarvam-M LLM  ->  Bulbul v3 TTS  ->  Speech Out

Covers:
  - Auto language detection from audio
  - Free Sarvam-M chat completion
  - Language-matched TTS response
  - Saving input/output audio and text transcript

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Usage:
  python 05_pipeline.py [input_audio.wav]
  Default: user_query.wav in current directory

The script:
  1. Transcribes the audio in the detected Indian language
  2. Sends the text to the free Sarvam-M LLM
  3. Converts the LLM reply to speech in the same language
  4. Saves the reply audio as agent_response.wav

Docs: https://docs.sarvam.ai
"""

import os
import sys
from pathlib import Path
from sarvamai import SarvamAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("ERROR: SARVAM_API_KEY environment variable not set.")

INPUT_AUDIO   = sys.argv[1] if len(sys.argv) > 1 else "user_query.wav"
OUTPUT_AUDIO  = "agent_response.wav"
LLM_MODEL     = "sarvam-m"   # free model; upgrade to sarvam-30b for quality
TTS_SPEAKER   = "priya"      # female voice; change as preferred

client = SarvamAI(api_subscription_key=API_KEY)


# ---------------------------------------------------------------------------
# Step 1: Speech to Text
# ---------------------------------------------------------------------------
def step_stt(audio_path: str) -> tuple[str, str]:
    """
    Transcribe audio and return (transcript, detected_language_code).
    language_code will be e.g. 'hi-IN', 'gu-IN', 'mr-IN'.
    """
    print("\n[Step 1] Speech -> Text (Saaras v3)")
    print(f"  Audio file : {audio_path}")

    with open(audio_path, "rb") as f:
        response = client.speech_to_text.transcribe(
            file=f,
            model="saaras:v3",
            mode="transcribe",   # return text in original language
            # language_code omitted -> auto-detect
        )

    transcript    = response.transcript
    detected_lang = getattr(response, "language_code", "hi-IN")

    print(f"  Detected   : {detected_lang}")
    print(f"  Transcript : {transcript}")
    return transcript, detected_lang


# ---------------------------------------------------------------------------
# Step 2: LLM Chat Completion
# ---------------------------------------------------------------------------
def step_llm(user_text: str, language_code: str) -> str:
    """
    Send the transcribed text to Sarvam-M (free) and return the reply.
    The system prompt instructs the model to reply in the detected language.
    """
    print("\n[Step 2] Text -> LLM Response (Sarvam-M -- free)")
    print(f"  Model      : {LLM_MODEL}")
    print(f"  User input : {user_text}")

    # Map language code to a human-readable language name for the prompt
    lang_names = {
        "hi-IN": "Hindi",
        "gu-IN": "Gujarati",
        "mr-IN": "Marathi",
        "bn-IN": "Bengali",
        "kn-IN": "Kannada",
        "ml-IN": "Malayalam",
        "pa-IN": "Punjabi",
        "te-IN": "Telugu",
        "ta-IN": "Tamil",
        "od-IN": "Odia",
        "en-IN": "English",
    }
    lang_name = lang_names.get(language_code, language_code)

    response = client.chat.completions(
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a helpful and friendly assistant. "
                    f"Always reply in {lang_name}. "
                    f"Keep your answer concise and clear."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        model=LLM_MODEL,
    )

    reply = response.choices[0].message.content
    print(f"  LLM reply  : {reply}")
    return reply


# ---------------------------------------------------------------------------
# Step 3: Text to Speech
# ---------------------------------------------------------------------------
def step_tts(text: str, language_code: str, output_path: str):
    """
    Convert the LLM reply to speech in the same language and save to file.
    Falls back to en-IN if Bulbul v3 does not support the detected language.
    """
    print("\n[Step 3] Text -> Speech (Bulbul v3)")

    # Bulbul v3 supports 11 languages; fall back to en-IN for others
    bulbul_supported = {
        "hi-IN", "gu-IN", "mr-IN", "bn-IN", "kn-IN",
        "ml-IN", "pa-IN", "te-IN", "ta-IN", "od-IN", "en-IN",
    }
    tts_lang = language_code if language_code in bulbul_supported else "en-IN"
    if tts_lang != language_code:
        print(f"  Note: Bulbul v3 does not support {language_code}. Using en-IN.")

    print(f"  Language   : {tts_lang}")
    print(f"  Speaker    : {TTS_SPEAKER}")
    print(f"  Text       : {text[:80]}{'...' if len(text) > 80 else ''}")

    response = client.text_to_speech.convert(
        target_language_code=tts_lang,
        text=text,
        model="bulbul:v3",
        speaker=TTS_SPEAKER,
        speech_sample_rate=16000,
        enable_preprocessing=True,
    )

    with open(output_path, "wb") as f:
        f.write(bytes(response.audios[0]))
    print(f"  Saved      : {output_path}")


# ---------------------------------------------------------------------------
# Save transcript log
# ---------------------------------------------------------------------------
def save_transcript(user_text: str, lang: str, reply: str):
    log_path = "pipeline_transcript.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Language detected : {lang}\n")
        f.write(f"User (STT)        : {user_text}\n")
        f.write(f"Agent (LLM)       : {reply}\n")
    print(f"\n  Transcript log saved -> {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Full Voice Pipeline")
    print("  STT (Saaras v3) -> LLM (Sarvam-M free) -> TTS (Bulbul v3)")
    print("Docs: https://docs.sarvam.ai")

    if not Path(INPUT_AUDIO).exists():
        print(
            f"\nERROR: Input audio file '{INPUT_AUDIO}' not found.\n"
            "Provide a WAV/MP3 file (under 30s) of Indian-language speech.\n"
            "Usage: python 05_pipeline.py <audio_file>"
        )
        sys.exit(1)

    # Run the three-step pipeline
    user_text, detected_lang = step_stt(INPUT_AUDIO)
    llm_reply                = step_llm(user_text, detected_lang)
    step_tts(llm_reply, detected_lang, OUTPUT_AUDIO)
    save_transcript(user_text, detected_lang, llm_reply)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Input audio    : {INPUT_AUDIO}")
    print(f"  Output audio   : {OUTPUT_AUDIO}")
    print(f"  Transcript log : pipeline_transcript.txt")
    print("=" * 60)
