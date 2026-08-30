"""
08_ocr_pipeline.py
==================
Sarvam AI - Document Intelligence (OCR) Pipeline
Covers:
  - Extract text from Indian-language PDFs (currently FREE)
  - Page-by-page structured output
  - Translate extracted text to English
  - Read the document aloud with Bulbul v3 TTS
  - Save extracted text, translated text, and audio to disk

Limits:
  - Max 10 pages per request
  - Rate limit: 10 req/min (same across all plans)
  - Uses the /v1/parse REST endpoint (not yet in SDK, called via httpx)

Install:  pip install sarvamai httpx python-dotenv
Set key:  export SARVAM_API_KEY="your_key_here"

Usage:
  python 08_ocr_pipeline.py [document.pdf] [source_language]
  Examples:
    python 08_ocr_pipeline.py gujarati_document.pdf gu-IN
    python 08_ocr_pipeline.py hindi_form.pdf hi-IN
  Defaults: gujarati_document.pdf  gu-IN

Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/document-intelligence
"""

import base64
import os
import sys
from pathlib import Path

import httpx
from sarvamai import SarvamAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_KEY = os.environ.get("SARVAM_API_KEY")
if not API_KEY:
    sys.exit("ERROR: SARVAM_API_KEY environment variable not set.")

PDF_FILE    = sys.argv[1] if len(sys.argv) > 1 else "gujarati_document.pdf"
SOURCE_LANG = sys.argv[2] if len(sys.argv) > 2 else "gu-IN"
OCR_URL     = "https://api.sarvam.ai/v1/parse"

client = SarvamAI(api_subscription_key=API_KEY)

# ---------------------------------------------------------------------------
# Step 1: OCR – extract text from PDF
# ---------------------------------------------------------------------------
def step_ocr(pdf_path: str) -> dict:
    """
    Send a PDF to the Sarvam Document Intelligence API.
    Returns the parsed JSON response with 'text' and 'pages' fields.
    """
    print("\n[Step 1] Document OCR (Sarvam Vision -- currently FREE)")
    print(f"  File       : {pdf_path}")

    with open(pdf_path, "rb") as f:
        pdf_b64 = base64.b64encode(f.read()).decode()

    response = httpx.post(
        OCR_URL,
        headers={"api-subscription-key": API_KEY},
        json={"file": pdf_b64, "file_type": "pdf"},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    full_text = data.get("text", "")
    pages     = data.get("pages", [])

    print(f"  Pages      : {len(pages)}")
    print(f"  Total chars: {len(full_text)}")
    print(f"  Preview    : {full_text[:200]}{'...' if len(full_text) > 200 else ''}")

    return data


# ---------------------------------------------------------------------------
# Step 2: Save extracted text to files
# ---------------------------------------------------------------------------
def step_save_extracted(data: dict, source_lang: str):
    """
    Save full text and page-by-page text to disk.
    """
    print("\n[Step 2] Saving Extracted Text")

    full_text = data.get("text", "")
    pages     = data.get("pages", [])

    # Save full text
    full_path = f"extracted_full_{source_lang.replace('-', '_')}.txt"
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    print(f"  Full text  -> {full_path}")

    # Save page-by-page
    if pages:
        pages_path = f"extracted_pages_{source_lang.replace('-', '_')}.txt"
        with open(pages_path, "w", encoding="utf-8") as f:
            for i, page in enumerate(pages, 1):
                page_text = page.get("text", "") if isinstance(page, dict) else str(page)
                f.write(f"=== Page {i} ===\n{page_text}\n\n")
        print(f"  Page-by-page -> {pages_path}")

    return full_text


# ---------------------------------------------------------------------------
# Step 3: Translate to English
# ---------------------------------------------------------------------------
def step_translate(text: str, source_lang: str) -> str:
    """
    Translate extracted text to English.
    Handles texts longer than typical API limits by chunking.
    """
    print("\n[Step 3] Translating to English")

    CHUNK_CHARS = 1500  # safe chunk size for translation API

    if len(text) <= CHUNK_CHARS:
        chunks = [text]
    else:
        # Split on sentence boundaries where possible
        chunks = []
        remaining = text
        while len(remaining) > CHUNK_CHARS:
            # Find last sentence-ending punctuation within limit
            split_at = remaining.rfind("।", 0, CHUNK_CHARS)
            if split_at == -1:
                split_at = remaining.rfind(".", 0, CHUNK_CHARS)
            if split_at == -1:
                split_at = CHUNK_CHARS
            chunks.append(remaining[:split_at + 1].strip())
            remaining = remaining[split_at + 1:].strip()
        if remaining:
            chunks.append(remaining)

    print(f"  Source lang: {source_lang}")
    print(f"  Chunks     : {len(chunks)}")

    translated_parts = []
    for i, chunk in enumerate(chunks, 1):
        response = client.text.translate(
            input=chunk,
            source_language_code=source_lang,
            target_language_code="en-IN",
            mode="formal",
        )
        translated_parts.append(response.translated_text)
        print(f"  Chunk {i}/{len(chunks)} translated ({len(chunk)} chars)")

    full_translation = " ".join(translated_parts)

    # Save translated text
    trans_path = "extracted_english.txt"
    with open(trans_path, "w", encoding="utf-8") as f:
        f.write(full_translation)
    print(f"  Saved      -> {trans_path}")
    print(f"  Preview    : {full_translation[:200]}...")

    return full_translation


# ---------------------------------------------------------------------------
# Step 4: Read the document aloud with TTS
# ---------------------------------------------------------------------------
def step_tts_readout(text: str, source_lang: str):
    """
    Convert the extracted text (first 500 chars) to speech using Bulbul v3.
    """
    print("\n[Step 4] Text-to-Speech Readout (Bulbul v3)")

    # Bulbul v3 supports 11 languages; fall back to en-IN for others
    bulbul_supported = {
        "hi-IN", "gu-IN", "mr-IN", "bn-IN", "kn-IN",
        "ml-IN", "pa-IN", "te-IN", "ta-IN", "od-IN", "en-IN",
    }
    tts_lang = source_lang if source_lang in bulbul_supported else "en-IN"

    # Choose an appropriate speaker for the language
    lang_speakers = {
        "gu-IN": "varun",
        "mr-IN": "kabir",
        "hi-IN": "anand",
        "bn-IN": "mani",
        "kn-IN": "gokul",
        "en-IN": "aditya",
    }
    speaker = lang_speakers.get(tts_lang, "anand")

    tts_text = text[:500]   # TTS for first 500 chars (demo)
    print(f"  Language   : {tts_lang}  Speaker: {speaker}")
    print(f"  Characters : {len(tts_text)}")

    response = client.text_to_speech.convert(
        target_language_code=tts_lang,
        text=tts_text,
        model="bulbul:v3",
        speaker=speaker,
        speech_sample_rate=16000,
        enable_preprocessing=True,
    )

    audio_path = f"document_readout_{tts_lang.replace('-', '_')}.wav"
    with open(audio_path, "wb") as f:
        f.write(bytes(response.audios[0]))
    print(f"  Saved      -> {audio_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Sarvam AI -- Document Intelligence Pipeline (OCR FREE)")
    print("  OCR (Sarvam Vision) -> Translate -> TTS (Bulbul v3)")
    print("Docs: https://docs.sarvam.ai/api-reference-docs/api-guides-tutorials/document-intelligence")

    if not Path(PDF_FILE).exists():
        print(
            f"\nERROR: PDF file '{PDF_FILE}' not found.\n"
            "Provide a PDF with Indian-language text (max 10 pages).\n"
            "Usage: python 08_ocr_pipeline.py <file.pdf> <lang_code>\n"
            "Example: python 08_ocr_pipeline.py gujarat_circular.pdf gu-IN"
        )
        sys.exit(1)

    print(f"\nPDF file    : {PDF_FILE}")
    print(f"Source lang : {SOURCE_LANG}")

    ocr_data      = step_ocr(PDF_FILE)
    extracted_text = step_save_extracted(ocr_data, SOURCE_LANG)

    if not extracted_text.strip():
        print("\nWARNING: No text extracted from PDF. Is it a scanned image?")
        sys.exit(0)

    translated_text = step_translate(extracted_text, SOURCE_LANG)
    step_tts_readout(extracted_text, SOURCE_LANG)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"  Extracted text (orig)  : extracted_full_{SOURCE_LANG.replace('-','_')}.txt")
    print(f"  Extracted (page-by-pg) : extracted_pages_{SOURCE_LANG.replace('-','_')}.txt")
    print(f"  Translated (English)   : extracted_english.txt")
    print(f"  Audio readout          : document_readout_{SOURCE_LANG.replace('-','_')}.wav")
    print("=" * 60)
