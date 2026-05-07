# core/transcriber.py
"""
Speech-to-text using OpenAI Whisper API.
No local model needed — sends audio to OpenAI and gets transcript back.
Works on Streamlit Cloud with no additional dependencies.
"""

import tempfile
import os
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float
    word_count: int
    words_per_minute: float


def transcribe_audio(
    filepath: str,
    api_key: str,
    model_size: str = "base"   # kept for API compatibility, ignored here
) -> TranscriptionResult:
    """
    Transcribe an audio file using OpenAI Whisper API.

    Args:
        filepath: Path to the audio file (.wav, .mp3, etc.)
        api_key: OpenAI API key
        model_size: Ignored — kept for backward compatibility

    Returns:
        TranscriptionResult with text and metadata
    """
    client = OpenAI(api_key=api_key)

    with open(filepath, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json"
        )

    text = response.text.strip()

    # Extract duration from verbose response if available
    try:
        duration = float(response.duration)
    except Exception:
        duration = 0.0

    # Detect language
    try:
        language = response.language or "en"
    except Exception:
        language = "en"

    word_count = len(text.split())
    wpm = (word_count / duration * 60) if duration > 0 else 0.0

    return TranscriptionResult(
        text=text,
        language=language,
        duration_seconds=duration,
        word_count=word_count,
        words_per_minute=round(wpm, 1)
    )
