#!/usr/bin/env python3
# setup_check.py
"""
Post-install verification and model download script.
Run this ONCE after pip install -r requirements.txt to:
  1. Download the spaCy language model
  2. Pre-download the Whisper base model
  3. Pre-download the sentence-transformers model
  4. Verify all imports work

Usage:
    python setup_check.py
"""

import sys


def check_import(module_name: str, package_hint: str = None) -> bool:
    try:
        __import__(module_name)
        print(f"  ✓ {module_name}")
        return True
    except ImportError as e:
        hint = f" (install: pip install {package_hint})" if package_hint else ""
        print(f"  ✗ {module_name}: {e}{hint}")
        return False


def main():
    print("\n" + "═" * 55)
    print("  INTERVIEW ANALYZER — SETUP CHECK")
    print("═" * 55)

    print("\n[1/5] Checking core imports...")
    all_ok = True
    all_ok &= check_import("whisper", "openai-whisper")
    all_ok &= check_import("spacy", "spacy")
    all_ok &= check_import("sentence_transformers", "sentence-transformers")
    all_ok &= check_import("gtts", "gTTS")
    all_ok &= check_import("streamlit", "streamlit")
    all_ok &= check_import("sounddevice", "sounddevice")
    all_ok &= check_import("soundfile", "soundfile")
    all_ok &= check_import("torch", "torch")

    if not all_ok:
        print("\n⚠  Some packages are missing. Run: pip install -r requirements.txt")
        sys.exit(1)

    print("\n[2/5] Downloading spaCy model (en_core_web_sm)...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print("  ✓ spaCy model downloaded.")
    else:
        # May already be installed
        try:
            import spacy
            spacy.load("en_core_web_sm")
            print("  ✓ spaCy model already available.")
        except OSError:
            print(f"  ✗ spaCy model download failed:\n{result.stderr}")
            sys.exit(1)

    print("\n[3/5] Pre-loading Whisper 'base' model...")
    try:
        import whisper
        model = whisper.load_model("base")
        print("  ✓ Whisper base model loaded.")
        del model
    except Exception as e:
        print(f"  ✗ Whisper model error: {e}")

    print("\n[4/5] Pre-loading sentence-transformers model...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        test_emb = model.encode("This is a test sentence.")
        print(f"  ✓ SentenceTransformer loaded. Embedding dim: {len(test_emb)}")
        del model
    except Exception as e:
        print(f"  ✗ SentenceTransformer error: {e}")

    print("\n[5/5] Testing gTTS...")
    try:
        from gtts import gTTS
        import tempfile, os
        tts = gTTS("Setup test.", lang="en")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tts.save(tmp.name)
        os.unlink(tmp.name)
        print("  ✓ gTTS working (internet connection confirmed).")
    except Exception as e:
        print(f"  ⚠ gTTS failed: {e}. Audio feedback requires internet access.")

    print("\n" + "═" * 55)
    print("  SETUP COMPLETE — Ready to run!")
    print("═" * 55)
    print("\nStart the app with:")
    print("  streamlit run ui/app.py")
    print("\nOr use the CLI:")
    print("  python run_cli.py --audio answer.wav --question 'Tell me about yourself.' --type hr\n")


if __name__ == "__main__":
    main()
