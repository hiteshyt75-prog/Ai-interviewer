# 🎙️ Audio-Based Interview Performance Analyzer

A fully functional, end-to-end system that records your spoken interview answer, transcribes it, and delivers explainable, multi-dimensional feedback — with spoken audio output.

---

## 📁 Directory Structure

```
interview_analyzer/
├── config/
│   ├── __init__.py
│   └── roles.py              # Role configs, HR competencies, scoring weights
├── core/
│   ├── __init__.py
│   ├── recorder.py           # Microphone recording (sounddevice)
│   ├── transcriber.py        # Speech-to-text (Whisper)
│   ├── nlp_analyzer.py       # NLP metrics (spaCy)
│   ├── semantic_analyzer.py  # Semantic relevance (sentence-transformers)
│   ├── scorer.py             # Weighted scoring engine
│   ├── feedback_generator.py # Natural language feedback
│   └── tts.py                # Text-to-speech (gTTS)
├── ui/
│   ├── __init__.py
│   └── app.py                # Streamlit web application
├── utils/
│   ├── __init__.py
│   └── pipeline.py           # End-to-end orchestration
├── output/
│   └── __init__.py
├── run_cli.py                 # Command-line interface
├── setup_check.py             # Post-install verification
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.9 – 3.11 (recommended: 3.10)
- Microphone (for recording mode)
- Internet connection (for gTTS audio feedback)
- ~2 GB disk space (models)

---

## 🚀 Setup Instructions

### Step 1 — Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `torch` is pulled in as a dependency of `whisper` and `sentence-transformers`.
> If you have a GPU, install the CUDA version of PyTorch first for faster transcription.

### Step 3 — Run setup check (downloads all models)

```bash
python setup_check.py
```

This will:
- Verify all package imports
- Download the spaCy `en_core_web_sm` model (~12 MB)
- Pre-cache the Whisper `base` model (~74 MB)
- Pre-cache `all-MiniLM-L6-v2` from sentence-transformers (~80 MB)
- Test gTTS internet connectivity

---

## ▶️ Running the App

### Streamlit UI (recommended)

```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

### Command-Line Interface

```bash
# HR behavioural question
python run_cli.py \
    --audio path/to/your_answer.wav \
    --question "Tell me about yourself." \
    --type hr

# Technical question with role
python run_cli.py \
    --audio path/to/your_answer.wav \
    --question "Explain the bias-variance tradeoff." \
    --type technical \
    --role data_scientist

# Skip audio feedback (offline mode)
python run_cli.py \
    --audio answer.wav \
    --question "Describe a conflict you resolved." \
    --type hr \
    --no-audio
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--audio` | required | Path to audio file |
| `--question` | required | Interview question |
| `--type` | `hr` | `hr` or `technical` |
| `--role` | `general` | Role key (technical only) |
| `--model` | `base` | Whisper model size |
| `--no-audio` | false | Skip TTS output |

---

## 🔬 How It Works

### Pipeline (5 steps)

```
Audio File
    │
    ▼ Step 1: Transcription
Whisper ASR  ──►  Raw text + duration + WPM
    │
    ▼ Step 2: NLP Analysis
spaCy  ──►  Sentence structure, filler words, STAR detection,
            named entities, vocabulary richness, passive voice
    │
    ▼ Step 3: Semantic Relevance
sentence-transformers  ──►  Cosine similarity vs. ideal answer bank
                            Keyword coverage for role-specific terms
    │
    ▼ Step 4: Scoring Engine
Weighted formula  ──►  Dimension scores + total (0–100) + grade (A–F)
    │
    ▼ Step 5: Feedback + TTS
Templates + gTTS  ──►  Textual feedback + spoken MP3 output
```

### Scoring Dimensions

**HR Interview:**

| Dimension | Weight | Basis |
|-----------|--------|-------|
| Relevance | 25% | Cosine similarity to ideal answer |
| Structure (STAR) | 20% | STAR components + transition words |
| Competency Coverage | 25% | HR keyword density per competency |
| Specificity | 20% | Quantified results + named entities + vocab |
| Fluency | 10% | Filler rate + speaking pace (WPM) |

**Technical Interview:**

| Dimension | Weight | Basis |
|-----------|--------|-------|
| Relevance | 20% | Cosine similarity to ideal answer |
| Technical Keywords | 25% | Role-specific keyword coverage |
| Depth | 25% | Length + vocabulary + sentence complexity |
| Structure | 15% | Transition words + logical flow |
| Fluency | 15% | Filler rate + speaking pace |

### Roles Supported (Technical)

- `software_engineer` — algorithms, OOP, APIs, databases
- `data_scientist` — ML, statistics, feature engineering
- `product_manager` — roadmaps, OKRs, user research
- `devops_engineer` — Kubernetes, CI/CD, IaC
- `general` — generic technical topics

---

## 📊 Example Output

```
══════════════════════════════════════════════════════
  RESULTS
══════════════════════════════════════════════════════

  SCORE : 71.4/100   Grade: B   [Good]

  TRANSCRIPTION (187 words, 142 WPM):
  "In my previous role at a fintech startup, I was tasked with
  improving the deployment pipeline which was taking over 2 hours..."

  DIMENSION SCORES:
  Relevance                      [████████████████░░░░] 78%
  Structure (STAR)               [██████████████░░░░░░] 70%
  Competency Coverage            [███████████████░░░░░] 74%
  Specificity & Examples         [█████████████░░░░░░░] 65%
  Fluency & Delivery             [██████████████████░░] 88%

  STRENGTHS:
    ✓ Relevance
    ✓ Fluency & Delivery

  AREAS TO IMPROVE:
    ✗ Specificity & Examples

  ACTION ITEMS:
  1. Add at least one specific metric or number to your answer.
  2. Use transition words (firstly, therefore, as a result) to structure your thinking.
```

---

## ⚠️ Troubleshooting

**`OSError: PortAudio not found`**
```bash
# macOS
brew install portaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev

# Then reinstall sounddevice
pip install sounddevice
```

**`OSError: [Errno -9999] Unanticipated host error`**
- Check microphone permissions in system settings.
- On macOS: System Preferences → Security & Privacy → Microphone.

**`spaCy model not found`**
```bash
python -m spacy download en_core_web_sm
```

**`gTTS connection error`**
- gTTS requires an active internet connection.
- Use `--no-audio` flag in CLI or uncheck "Generate Audio Feedback" in UI.

**`torch not installed`**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 📝 Notes

- **No AI magic**: All scoring is based on deterministic, measurable linguistic features. No "confidence detection" or unsupported claims.
- **Privacy**: Audio is processed locally. gTTS sends text to Google's servers only for speech synthesis.
- **Model size vs. speed**: `tiny` Whisper is fastest (~1s); `small` is more accurate. `base` is the best default balance.
- **Language**: Whisper auto-detects language. NLP analysis is English-only (spaCy model).
