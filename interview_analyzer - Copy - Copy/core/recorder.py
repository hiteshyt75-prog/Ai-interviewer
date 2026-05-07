# core/recorder.py
import sounddevice as sd
import soundfile as sf
import tempfile
import os

SAMPLE_RATE = 16000
CHANNELS = 1


def record_audio(duration: int, output_path: str = None) -> str:
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    audio_data = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32"
    )
    sd.wait()
    sf.write(output_path, audio_data, SAMPLE_RATE)
    return output_path


def validate_audio_file(filepath: str) -> bool:
    if not os.path.exists(filepath):
        return False
    try:
        with sf.SoundFile(filepath):
            return True
    except Exception:
        return False
