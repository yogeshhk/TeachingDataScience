"""
09_call_analytics.py
====================
Sarvam AI - Batch Call Analytics Pipeline
Covers:
  - Batch STT API (up to 20 files, up to 1 hour each)
  - Diarization (speaker labels: SPEAKER_00, SPEAKER_01)
  - Parsing diarized JSON output into readable conversation
  - Speaker talk-time calculation
  - LLM analysis per call: agent vs customer, sentiment, resolution
  - Concise summary generation
  - Q&A against transcripts
  - Saving all outputs to structured files

Billing note:
  STT Batch with diarization: Rs. 45/hour of audio
  LLM (sarvam-m): FREE

Install:  pip install sarvamai pydub python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Usage:
  python 09_call_analytics.py call1.mp3 call2.mp3 ...
  If no files given, prints usage and exits.

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview
"""

import json
import os
import sys
import time
import textwrap
from datetime import datetime
from pathlib import Path

from sarvamai import SarvamAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("ERROR: SARVAM_API_KEY environment variable not set.")

if len(sys.argv) < 2:
    print(
        "Usage: python 09_call_analytics.py call1.mp3 [call2.mp3 ...]\n"
        "Provide one or more audio files (WAV/MP3, up to 1 hour each, max 20 files)."
    )
    sys.exit(1)

AUDIO_FILES  = sys.argv[1:]
OUTPUT_DIR   = Path("call_analytics_output")
OUTPUT_DIR.mkdir(exist_ok=True)
LLM_MODEL    = "sarvam-m"   # free

client = SarvamAI(api_subscription_key=API_KEY)

# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------
ANALYSIS_PROMPT = """
Analyse this call transcription thoroughly:

TRANSCRIPTION:
{transcription}

Answer each point:
1. Which speaker is the AGENT and which is the CUSTOMER?
2. Is the customer new or existing?
3. What problem or query did the customer raise?
4. What product or service was discussed?
5. How did the agent respond and resolve the issue?
6. Was the customer satisfied at the end?
7. What was the overall sentiment (positive / negative / neutral)?
8. Were competitors mentioned? Any upsell/cross-sell opportunity?
9. Summary of the resolution (successful / partial / unresolved).

Format your answer with clear headings and bullet points.
"""

SUMMARY_PROMPT = """
Based on this call analysis, give a 1-line answer for each:

{analysis}

1. Agent name or ID (if mentioned)
2. Customer type (new / existing)
3. Main issue (3-5 words)
4. Service/product discussed
5. Agent effectiveness (good / average / poor)
6. Customer satisfaction (satisfied / neutral / dissatisfied)
7. Sentiment
8. Resolution status (resolved / partial / unresolved)
"""


# ---------------------------------------------------------------------------
# Step 1: Batch transcription with diarization
# ---------------------------------------------------------------------------
def step_batch_transcribe(audio_paths: list[str]) -> Path:
    """
    Submit audio files to Batch STT API with diarization.
    Returns the output directory containing JSON transcription files.
    """
    print("\n[Step 1] Batch Transcription with Diarization")
    print(f"  Files      : {len(audio_paths)}")
    for p in audio_paths:
        print(f"    - {p}")

    job = client.speech_to_text_translate_job.create_job(
        model="saaras:v3",
        mode="translate",        # translate all speech to English
        with_diarization=True,   # label SPEAKER_00, SPEAKER_01 etc.
    )

    print(f"\n  Job ID     : {job.job_id}")
    print("  Uploading files...")

    job.upload_files(file_paths=audio_paths, timeout=300)
    job.start()

    print("  Job started. Waiting for completion...")
    print("  (Polling with 5s delay as recommended by Sarvam docs)")

    while not job.is_complete() and not job.is_failed():
        time.sleep(5)   # min 5ms recommended; 5s is safe for polling
        print("  ...", end="", flush=True)

    print()

    if job.is_failed():
        sys.exit("ERROR: Batch job failed. Check audio files and try again.")

    transcription_dir = OUTPUT_DIR / f"transcriptions_{job.job_id}"
    transcription_dir.mkdir(parents=True, exist_ok=True)
    job.download_outputs(output_dir=str(transcription_dir))

    json_files = list(transcription_dir.glob("*.json"))
    print(f"  Downloaded : {len(json_files)} transcription file(s) -> {transcription_dir}")

    return transcription_dir


# ---------------------------------------------------------------------------
# Step 2: Parse diarized JSON into readable conversation
# ---------------------------------------------------------------------------
def step_parse_transcriptions(transcription_dir: Path) -> dict:
    """
    Parse all JSON transcription files.
    Returns a dict keyed by file stem with conversation text and speaker timing.
    """
    print("\n[Step 2] Parsing Diarized Transcriptions")

    results = {}

    for json_file in transcription_dir.glob("*.json"):
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)

        diarized = data.get("diarized_transcript", {}).get("entries", [])
        speaker_times = {}
        lines = []

        if diarized:
            for entry in diarized:
                speaker  = entry.get("speaker_id", "UNKNOWN")
                text     = entry.get("transcript", "")
                start    = entry.get("start_time_seconds", 0)
                end      = entry.get("end_time_seconds",   0)

                lines.append(f"{speaker}: {text}")
                duration = end - start
                speaker_times[speaker] = speaker_times.get(speaker, 0.0) + duration
        else:
            # Fallback: no diarization in output
            raw = data.get("transcript", "")
            lines = [f"SPEAKER_00: {raw}"]
            speaker_times = {"SPEAKER_00": 0.0}

        conversation_text = "\n".join(lines)

        # Save conversation to .txt
        conv_path = transcription_dir / f"{json_file.stem}_conversation.txt"
        with open(conv_path, "w", encoding="utf-8") as f:
            f.write(conversation_text)

        # Save timing to JSON
        timing_path = transcription_dir / f"{json_file.stem}_timing.json"
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(speaker_times, f, indent=2)

        results[json_file.stem] = {
            "conversation_text": conversation_text,
            "conversation_path": str(conv_path),
            "timing_path":       str(timing_path),
            "speaker_times":     speaker_times,
        }

        print(f"\n  File: {json_file.name}")
        print(f"  Speakers: {list(speaker_times.keys())}")
        for spk, secs in speaker_times.items():
            print(f"    {spk}: {secs:.1f}s talk-time")
        print(f"  Preview:\n    {conversation_text[:200]}...")

    return results


# ---------------------------------------------------------------------------
# Step 3: LLM analysis per call
# ---------------------------------------------------------------------------
def step_analyse_calls(parsed: dict, transcription_dir: Path) -> dict:
    """
    Send each conversation to the free Sarvam-M LLM for structured analysis.
    """
    print("\n[Step 3] LLM Analysis per Call (sarvam-m -- free)")

    analysis_results = {}

    for stem, data in parsed.items():
        conversation = data["conversation_text"]
        prompt = textwrap.dedent(ANALYSIS_PROMPT.format(transcription=conversation))

        print(f"\n  Analysing: {stem}")

        response = client.chat.completions(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a call analytics expert for a customer support team. "
                        "Provide structured, actionable insights."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=LLM_MODEL,
        )

        analysis = response.choices[0].message.content.strip()

        # Save analysis
        analysis_path = transcription_dir / f"{stem}_analysis.txt"
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(analysis)

        print(f"  Saved -> {analysis_path}")
        print(f"  Preview:\n    {analysis[:300]}...")

        analysis_results[stem] = {
            **data,
            "analysis":      analysis,
            "analysis_path": str(analysis_path),
        }

    return analysis_results


# ---------------------------------------------------------------------------
# Step 4: Answer a custom question against transcripts
# ---------------------------------------------------------------------------
def step_answer_question(analysis_results: dict, question: str, transcription_dir: Path):
    """
    Answer a user-defined question based on each call transcript.
    """
    print(f"\n[Step 4] Q&A Against Transcripts")
    print(f"  Question: {question}")

    for stem, data in analysis_results.items():
        prompt = (
            f"Based on this call transcript, answer the question:\n\n"
            f"TRANSCRIPT:\n{data['conversation_text']}\n\n"
            f"QUESTION: {question}"
        )

        response = client.chat.completions(
            messages=[
                {"role": "system", "content": "Answer questions about call transcripts concisely."},
                {"role": "user",   "content": prompt},
            ],
            model=LLM_MODEL,
        )

        answer = response.choices[0].message.content.strip()
        qa_path = transcription_dir / f"{stem}_qa.txt"
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n\nAnswer:\n{answer}")

        print(f"\n  [{stem}] {answer[:200]}...")
        print(f"  Saved -> {qa_path}")


# ---------------------------------------------------------------------------
# Step 5: Generate summary report
# ---------------------------------------------------------------------------
def step_generate_summary(analysis_results: dict):
    """
    Generate a concise summary report across all calls.
    """
    print("\n[Step 5] Generating Summary Report")

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = OUTPUT_DIR / f"summary_{timestamp}.txt"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CALL ANALYTICS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated : {datetime.now()}\n")
        f.write(f"Total calls: {len(analysis_results)}\n")
        f.write("=" * 60 + "\n\n")

        for stem, data in analysis_results.items():
            analysis = data.get("analysis", "")
            if not analysis:
                continue

            prompt = textwrap.dedent(SUMMARY_PROMPT.format(analysis=analysis))

            response = client.chat.completions(
                messages=[
                    {"role": "system", "content": "Provide concise 1-line answers."},
                    {"role": "user",   "content": prompt},
                ],
                model=LLM_MODEL,
            )
            concise = response.choices[0].message.content.strip()

            f.write(f"Call: {stem}\n")
            f.write("-" * 40 + "\n")
            f.write(concise + "\n\n")

            # Speaker talk-time
            f.write("Speaker talk-time:\n")
            for spk, secs in data.get("speaker_times", {}).items():
                f.write(f"  {spk}: {secs:.1f}s\n")
            f.write("\n")

    print(f"  Summary -> {summary_path}")
    return summary_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Batch Call Analytics Pipeline")
    print("  Batch STT (diarization) -> Parse -> LLM Analysis -> Summary")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview")
    print(f"\nProcessing {len(AUDIO_FILES)} audio file(s):")
    for f in AUDIO_FILES:
        if not Path(f).exists():
            sys.exit(f"ERROR: File not found: {f}")
        print(f"  {f}")

    transcription_dir = step_batch_transcribe(AUDIO_FILES)
    parsed            = step_parse_transcriptions(transcription_dir)
    analysis_results  = step_analyse_calls(parsed, transcription_dir)

    # Custom question against all transcripts
    step_answer_question(
        analysis_results,
        question="Was the customer's issue resolved by the end of the call?",
        transcription_dir=transcription_dir,
    )

    summary_path = step_generate_summary(analysis_results)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Transcriptions : {transcription_dir}/")
    print(f"  Summary report : {summary_path}")
    print("=" * 60)
