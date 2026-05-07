# core/tts.py
import os
import re
import tempfile
from gtts import gTTS


def text_to_speech(text: str, lang: str = "en") -> str:
    """Convert text to speech, save as MP3, return file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.close()
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(tmp.name)
    return tmp.name


def chunk_and_speak(text: str, max_chars: int = 400, lang: str = "en") -> str:
    """Handle long text by chunking into segments."""
    if len(text) <= max_chars:
        return text_to_speech(text, lang=lang)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= max_chars:
            current += " " + s
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s
    if current.strip():
        chunks.append(current.strip())

    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_files.append(text_to_speech(chunk, lang=lang))

    try:
        from pydub import AudioSegment
        combined = AudioSegment.empty()
        for cf in chunk_files:
            combined += AudioSegment.from_mp3(cf)
        out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        out.close()
        combined.export(out.name, format="mp3")
        for cf in chunk_files:
            os.unlink(cf)
        return out.name
    except Exception:
        return chunk_files[0] if chunk_files else text_to_speech(text)
