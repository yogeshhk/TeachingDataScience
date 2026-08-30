"""
03_stt_streaming.py
===================
Sarvam AI - Real-Time Speech-to-Text via WebSocket (Saaras v3)
Covers:
  - AsyncSarvamAI client
  - WebSocket streaming for live transcription
  - Chunked audio push (simulating a live mic stream from a file)
  - Receiving incremental transcript events as they arrive

Install:  pip install sarvamai python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Audio constraint: WebSocket STT ONLY accepts WAV or raw PCM formats
                  (pcm_s16le, pcm_l16, pcm_raw) at 16 kHz sample rate.
                  MP3/AAC are NOT supported for streaming.

Usage:  python 03_stt_streaming.py [audio_file.wav]
        Default file: live_audio.wav in current directory.

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview
"""

import asyncio
import os
import sys
from pathlib import Path
from sarvamai import AsyncSarvamAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("ERROR: SARVAM_API_KEY environment variable not set.")

AUDIO_FILE = sys.argv[1] if len(sys.argv) > 1 else "live_audio.wav"
CHUNK_SIZE  = 4096   # bytes per push -- simulates a live mic buffer
LANGUAGE    = "hi-IN"  # set to "unknown" for auto-detection


# ---------------------------------------------------------------------------
# Core streaming function
# ---------------------------------------------------------------------------
async def stream_transcription(audio_path: str):
    """Stream a WAV file to Saaras v3 WebSocket API and print transcripts."""

    if not Path(audio_path).exists():
        print(
            f"ERROR: '{audio_path}' not found.\n"
            "Provide a 16kHz mono WAV file.\n"
            "Usage: python 03_stt_streaming.py <audio_file.wav>"
        )
        return

    client = AsyncSarvamAI(api_subscription_key=API_KEY)

    print(f"Opening WebSocket stream for: {audio_path}")
    print(f"Language: {LANGUAGE}  |  Chunk size: {CHUNK_SIZE} bytes\n")

    async with client.speech_to_text.stream(
        model="saaras:v3",
        mode="transcribe",
        language_code=LANGUAGE,   # omit / "unknown" for auto-detect
    ) as stream:

        # Push audio chunks (mimics a live microphone feed)
        total_chunks = 0
        with open(audio_path, "rb") as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                await stream.send_audio(chunk)
                total_chunks += 1

        print(f"Pushed {total_chunks} chunks. Signalling end of stream...\n")
        await stream.end_stream()

        # Receive incremental transcript events
        print("Transcript events:")
        full_transcript = []
        async for event in stream:
            segment = getattr(event, "transcript", "")
            if segment:
                print(f"  [partial] {segment}")
                full_transcript.append(segment)

    print("\n" + "-" * 50)
    print("Full transcript:")
    print(" ".join(full_transcript))


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
def print_streaming_notes():
    print("\n" + "=" * 60)
    print("WebSocket STT -- Key Constraints")
    print("=" * 60)
    print("  Accepted formats  : WAV, pcm_s16le, pcm_l16, pcm_raw")
    print("  Sample rate       : 16 kHz (16,000 Hz)")
    print("  NOT supported     : MP3, AAC, OGG, FLAC for streaming")
    print("  Concurrent limit  : 20 WebSocket connections (Starter plan)")
    print("  Use case          : live voice bots, real-time captions")
    print("  For recorded files: use REST API (02_stt_modes_demo.py)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Real-Time STT Streaming Demo (Saaras v3)")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/speech-to-text/overview")

    print_streaming_notes()
    print()

    asyncio.run(stream_transcription(AUDIO_FILE))

    print("\nDone.")
