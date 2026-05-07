# core/transcriber.py
import whisper
import soundfile as sf
import os
from dataclasses import dataclass
from typing import Optional

_model_cache = {}


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration_seconds: float
    word_count: int
    words_per_minute: float


def load_model(model_size: str = "base") -> whisper.Whisper:
    if model_size not in _model_cache:
        print(f"[Transcriber] Loading Whisper '{model_size}'...")
        _model_cache[model_size] = whisper.load_model(model_size)
    return _model_cache[model_size]


def transcribe_audio(filepath: str, model_size: str = "base") -> TranscriptionResult:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    model = load_model(model_size)
    result = model.transcribe(filepath)
    text = result["text"].strip()

    try:
        with sf.SoundFile(filepath) as f:
            duration = len(f) / f.samplerate
    except Exception:
        duration = 0.0

    word_count = len(text.split())
    wpm = (word_count / duration * 60) if duration > 0 else 0.0

    return TranscriptionResult(
        text=text,
        language=result.get("language", "en"),
        duration_seconds=duration,
        word_count=word_count,
        words_per_minute=round(wpm, 1)
    )
