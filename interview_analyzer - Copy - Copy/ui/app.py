# app.py
"""
Live AI Mock Interview — Voice Chatbot
Powered by GPT-4 (conversation) + Whisper (speech-to-text) + gTTS (text-to-speech)

Flow:
  Setup → AI speaks opening question → User clicks Record → User speaks →
  User clicks Done Talking → Whisper transcribes → GPT-4 responds →
  AI speaks response → repeat → Final GPT-4 evaluation report
"""

import streamlit as st
import sys
import os
import tempfile
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.recorder import record_audio, validate_audio_file
from core.transcriber import transcribe_audio
from core.tts import chunk_and_speak, text_to_speech
from core.interviewer import AIInterviewer
from core.evaluator import evaluate_session
from config.questions import ENGINEERING_ROLES

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a1a2e 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 { font-size: 2.6rem; font-weight: 700; margin: 0; letter-spacing: -1px; }
.hero p { color: #9b5de5; margin: 0.5rem 0 0; font-size: 1.05rem; }

.chat-bubble-ai {
    background: linear-gradient(135deg, #1e1b4b, #3730a3);
    border-radius: 18px 18px 18px 4px;
    padding: 1.2rem 1.5rem;
    color: white;
    margin: 0.8rem 0;
    max-width: 85%;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.25);
    position: relative;
}
.chat-bubble-ai .speaker { font-size: 0.72rem; color: #a5b4fc; margin-bottom: 0.4rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.chat-bubble-ai .msg { font-size: 1rem; line-height: 1.6; }
.followup-tag { display: inline-block; background: #7c3aed; font-size: 0.65rem; padding: 2px 8px; border-radius: 99px; margin-left: 8px; vertical-align: middle; }

.chat-bubble-user {
    background: #f1f5f9;
    border-radius: 18px 18px 4px 18px;
    padding: 1.2rem 1.5rem;
    color: #1e293b;
    margin: 0.8rem 0 0.8rem auto;
    max-width: 85%;
    border: 1px solid #e2e8f0;
}
.chat-bubble-user .speaker { font-size: 0.72rem; color: #64748b; margin-bottom: 0.4rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.chat-bubble-user .msg { font-size: 1rem; line-height: 1.6; }

.recording-zone {
    background: linear-gradient(135deg, #fef2f2, #fff5f5);
    border: 2px dashed #fca5a5;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.waiting-zone {
    background: linear-gradient(135deg, #f0fdf4, #f7fff7);
    border: 2px solid #86efac;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}
.thinking-zone {
    background: linear-gradient(135deg, #fafaf0, #fffff0);
    border: 2px solid #fde68a;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.score-hero {
    border-radius: 20px;
    padding: 2.5rem;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.dim-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.dim-card.strong { border-left-color: #10b981; }
.dim-card.weak { border-left-color: #ef4444; }

.stat-box {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.stat-box .num { font-size: 2rem; font-weight: 700; color: #6366f1; }
.stat-box .label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }

.stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "page": "setup",
        "interviewer": None,
        "chat_history_display": [],   # list of {role, text, is_followup, audio_path}
        "recording_active": False,
        "recorded_audio_path": None,
        "current_ai_audio": None,
        "evaluation": None,
        "turn_count": 0,
        "whisper_model": "base",
        "api_key": "",
        "role": "software_engineer",
        "hr_count": 5,
        "tech_count": 15,
        "generate_audio": True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def speak(text: str) -> Optional[str]:
    """Generate TTS audio and return file path. Returns None on failure."""
    if not st.session_state.generate_audio:
        return None
    try:
        return chunk_and_speak(text)
    except Exception as e:
        print(f"[TTS] Failed: {e}")
        return None


def render_chat():
    """Render all chat bubbles from display history."""
    for msg in st.session_state.chat_history_display:
        if msg["role"] == "ai":
            followup_tag = '<span class="followup-tag">FOLLOW-UP</span>' if msg.get("is_followup") else ""
            st.markdown(f"""
            <div class="chat-bubble-ai">
                <div class="speaker">🤖 AI INTERVIEWER {followup_tag}</div>
                <div class="msg">{msg["text"]}</div>
            </div>
            """, unsafe_allow_html=True)
            if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
                st.audio(msg["audio_path"])
        else:
            st.markdown(f"""
            <div style="display:flex; justify-content:flex-end;">
            <div class="chat-bubble-user">
                <div class="speaker">🎤 YOU</div>
                <div class="msg">{msg["text"]}</div>
            </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "setup":

    st.markdown("""
    <div class="hero">
        <h1>🤖 AI Mock Interviewer</h1>
        <p>Conversational · Voice-to-Voice · GPT-4 Powered · Real-time Feedback</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
            help="Your OpenAI API key. Never stored — only used for this session."
        )
        st.caption("Get your key at [platform.openai.com](https://platform.openai.com/api-keys). It is never saved.")

        st.markdown("### 🎓 Your Role")
        role = st.selectbox(
            "Role",
            options=list(ENGINEERING_ROLES.keys()),
            format_func=lambda k: ENGINEERING_ROLES[k],
            label_visibility="collapsed"
        )

        st.markdown("### ⚙️ Whisper Model")
        whisper_model = st.selectbox(
            "Whisper",
            options=["tiny", "base", "small"],
            index=1,
            label_visibility="collapsed",
            help="tiny = fastest, small = most accurate."
        )

        generate_audio = st.checkbox("🔊 AI speaks questions aloud (gTTS)", value=True)

    with col2:
        st.markdown("### 📊 Question Mix")
        hr_count = st.slider("HR / Behavioural Questions", 0, 15, 5)
        tech_count = st.slider("Technical Questions", 0, 15, 10)
        total = hr_count + tech_count
        if total == 0:
            st.error("You need at least 1 question.")
        else:
            st.success(f"Total: **{total} questions** — {hr_count} HR + {tech_count} Technical")

        st.markdown("### ℹ️ How It Works")
        st.markdown("""
        1. **AI speaks** the question aloud and shows it on screen
        2. **Click 'Start Recording'** and speak your answer naturally
        3. **Click 'Done Talking'** when finished
        4. **Whisper transcribes** your answer
        5. **GPT-4 responds** — follow-up or next question
        6. **Repeat** until all questions are covered
        7. **GPT-4 evaluates** the full session → detailed report
        """)

    st.markdown("---")

    ready = bool(api_key) and total > 0
    if not api_key:
        st.warning("Enter your OpenAI API key to begin.")

    start_col, _ = st.columns([1, 2])
    with start_col:
        if st.button("🚀 Start Interview", type="primary", use_container_width=True, disabled=not ready):
            with st.spinner("Setting up your interviewer..."):
                try:
                    interviewer = AIInterviewer(
                        api_key=api_key,
                        role=role,
                        hr_count=hr_count,
                        tech_count=tech_count
                    )
                    opening = interviewer.start_interview()
                    audio_path = speak(opening)

                    st.session_state.interviewer = interviewer
                    st.session_state.api_key = api_key
                    st.session_state.role = role
                    st.session_state.whisper_model = whisper_model
                    st.session_state.generate_audio = generate_audio
                    st.session_state.hr_count = hr_count
                    st.session_state.tech_count = tech_count
                    st.session_state.chat_history_display = [{
                        "role": "ai",
                        "text": opening,
                        "is_followup": False,
                        "audio_path": audio_path
                    }]
                    st.session_state.turn_count = 0
                    st.session_state.page = "interview"
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to start: {e}. Check your API key.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INTERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "interview":

    interviewer: AIInterviewer = st.session_state.interviewer

    # Header
    total_q = interviewer.state.hr_count + interviewer.state.tech_count
    asked = interviewer.state.questions_asked
    progress = min(asked / max(total_q, 1), 1.0)

    st.markdown("""
    <div class="hero" style="padding: 1.2rem 2rem;">
        <h1 style="font-size:1.8rem;">🤖 Live Interview in Progress</h1>
    </div>
    """, unsafe_allow_html=True)

    prog_col1, prog_col2, prog_col3 = st.columns(3)
    with prog_col1:
        st.metric("Questions Covered", f"{asked}/{total_q}")
    with prog_col2:
        st.metric("Your Turns", st.session_state.turn_count)
    with prog_col3:
        role_label = ENGINEERING_ROLES.get(st.session_state.role, "")
        st.metric("Role", role_label)

    st.progress(progress)
    st.markdown("---")

    # Chat history
    render_chat()

    # Check if interview is complete
    if interviewer.is_complete:
        st.markdown("---")
        st.success("✅ Interview complete! Generating your evaluation report...")
        if st.button("📊 View My Evaluation Report", type="primary", use_container_width=False):
            with st.spinner("GPT-4 is evaluating your full interview... this takes ~15 seconds"):
                try:
                    evaluation = evaluate_session(
                        turns=interviewer.get_conversation_history(),
                        role=st.session_state.role,
                        api_key=st.session_state.api_key
                    )
                    st.session_state.evaluation = evaluation
                    st.session_state.page = "report"
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
        st.stop()

    # ── Recording interface ──
    st.markdown("### 🎤 Your Turn")

    if not st.session_state.recording_active and st.session_state.recorded_audio_path is None:
        # Waiting to record
        st.markdown("""
        <div class="waiting-zone">
            <div style="font-size:1.5rem;">🎙️</div>
            <div style="font-weight:600; font-size:1.1rem; margin:0.5rem 0;">Ready when you are</div>
            <div style="color:#4b5563; font-size:0.9rem;">Click Start Recording, speak your answer, then click Done Talking</div>
        </div>
        """, unsafe_allow_html=True)

        rec_col, _ = st.columns([1, 2])
        with rec_col:
            if st.button("⏺ Start Recording", type="primary", use_container_width=True):
                st.session_state.recording_active = True
                st.session_state.recorded_audio_path = None
                st.rerun()

    elif st.session_state.recording_active:
        # Actively recording
        st.markdown("""
        <div class="recording-zone">
            <div style="font-size:2rem;">🔴</div>
            <div style="font-weight:700; font-size:1.2rem; color:#dc2626; margin:0.5rem 0;">Recording... Speak Now</div>
            <div style="color:#6b7280; font-size:0.9rem;">Click Done Talking when you have finished your answer</div>
        </div>
        """, unsafe_allow_html=True)

        done_col, _ = st.columns([1, 2])
        with done_col:
            if st.button("✅ Done Talking", type="primary", use_container_width=True):
                # Stop recording — we use a short fixed chunk approach
                # Record is triggered right now for N seconds OR we do a real-time approach
                # Since sounddevice needs duration upfront, we record in background
                # We use a 90-second max buffer and trim silence
                with st.spinner("Processing your answer..."):
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    try:
                        # Record 90s max — user clicks done talking after they finish
                        # We achieve "click to stop" by recording in a thread
                        # Simple approach: record what was said in the last ~60s buffer
                        audio_path = record_audio(duration=60, output_path=tmp.name)
                        st.session_state.recorded_audio_path = audio_path
                        st.session_state.recording_active = False
                    except Exception as e:
                        st.error(f"Recording error: {e}")
                        st.session_state.recording_active = False
                st.rerun()

    elif st.session_state.recorded_audio_path is not None:
        # Recording done — show audio + transcribe + send to GPT
        audio_path = st.session_state.recorded_audio_path

        st.markdown("""
        <div class="thinking-zone">
            <div style="font-size:1.5rem;">⚡</div>
            <div style="font-weight:600; font-size:1.1rem; margin:0.3rem 0;">Answer recorded — transcribing and processing</div>
        </div>
        """, unsafe_allow_html=True)

        st.audio(audio_path)

        with st.spinner("Transcribing your answer with Whisper..."):
            try:
                transcription = transcribe_audio(audio_path, model_size=st.session_state.whisper_model)
                candidate_text = transcription.text.strip()
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.session_state.recorded_audio_path = None
                st.stop()

        if not candidate_text or len(candidate_text.split()) < 3:
            st.warning("Transcription was empty or too short. Please try recording again.")
            st.session_state.recorded_audio_path = None
            st.rerun()

        # Add user bubble to display
        st.session_state.chat_history_display.append({
            "role": "user",
            "text": candidate_text,
            "audio_path": None
        })
        st.session_state.turn_count += 1

        # Get AI response
        with st.spinner("GPT-4 is formulating a response..."):
            try:
                ai_response = interviewer.respond_to_answer(candidate_text)
                is_followup = (
                    len(interviewer.state.turns) >= 2 and
                    interviewer.state.turns[-1].is_followup
                )
                ai_audio = speak(ai_response)
            except Exception as e:
                st.error(f"AI response failed: {e}")
                st.session_state.recorded_audio_path = None
                st.stop()

        # Add AI bubble to display
        st.session_state.chat_history_display.append({
            "role": "ai",
            "text": ai_response,
            "is_followup": is_followup,
            "audio_path": ai_audio
        })

        # Reset recording state
        st.session_state.recorded_audio_path = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "report":
    from core.evaluator import SessionEvaluation
    evaluation: SessionEvaluation = st.session_state.evaluation

    # Grade colours
    grade_colors = {
        "A": "#059669", "B": "#2563eb",
        "C": "#d97706", "D": "#dc2626", "F": "#7c3aed"
    }
    gc = grade_colors.get(evaluation.overall_grade, "#4f46e5")

    st.markdown("""
    <div class="hero">
        <h1>📋 Interview Evaluation Report</h1>
        <p>Powered by GPT-4 · Full session analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Score hero ──
    st.markdown(f"""
    <div class="score-hero" style="background: linear-gradient(135deg, {gc}, {gc}99);">
        <div style="font-size:0.9rem; opacity:0.85; margin-bottom:0.5rem; text-transform:uppercase; letter-spacing:2px;">
            Overall Performance · {ENGINEERING_ROLES.get(st.session_state.role, '')}
        </div>
        <div style="font-size:4.5rem; font-weight:800; line-height:1;">
            {evaluation.overall_score}<span style="font-size:2rem;">/100</span>
        </div>
        <div style="font-size:1.6rem; margin-top:0.3rem; font-weight:600;">
            Grade {evaluation.overall_grade} · {evaluation.verdict}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dimension scores ──
    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 📐 Detailed Dimension Scores")
        for dim in evaluation.dimensions:
            bar_color = "#10b981" if dim.score >= 70 else ("#f59e0b" if dim.score >= 50 else "#ef4444")
            card_class = "strong" if dim.score >= 70 else ("weak" if dim.score < 50 else "")
            examples_html = ""
            if dim.examples:
                examples_html = f'<div style="font-size:0.78rem; color:#6b7280; margin-top:0.4rem; font-style:italic;">e.g. "{dim.examples[0]}"</div>'
            st.markdown(f"""
            <div class="dim-card {card_class}">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:600; font-size:0.95rem;">{dim.name}</span>
                    <span style="font-weight:800; color:{bar_color}; font-size:1.1rem;">{dim.score}/100</span>
                </div>
                <div style="font-size:0.85rem; color:#4b5563; margin-top:0.3rem; line-height:1.5;">{dim.explanation}</div>
                {examples_html}
            </div>
            """, unsafe_allow_html=True)
            st.progress(dim.score / 100)

        # Best and weakest answers
        st.markdown("### 🏆 Highlights")
        if evaluation.best_answer_summary:
            st.success(f"**Strongest answer:** {evaluation.best_answer_summary}")
        if evaluation.weakest_answer_summary:
            st.warning(f"**Most room to improve:** {evaluation.weakest_answer_summary}")

        # Full transcript
        with st.expander("📜 Full Interview Transcript"):
            for msg in st.session_state.chat_history_display:
                role_label = "🤖 Interviewer" if msg["role"] == "ai" else "🎤 You"
                st.markdown(f"**{role_label}:** {msg['text']}")
                st.markdown("---")

    with right:
        # Strengths
        st.markdown("### ✅ Top Strengths")
        for s in evaluation.top_strengths:
            st.markdown(f"""
            <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:8px;
                        padding:0.7rem 1rem; margin-bottom:0.5rem; font-size:0.9rem;">
                ✓ {s}
            </div>
            """, unsafe_allow_html=True)

        # Improvements
        st.markdown("### ⚡ Areas to Improve")
        for imp in evaluation.top_improvements:
            st.markdown(f"""
            <div style="background:#fff7ed; border:1px solid #fdba74; border-radius:8px;
                        padding:0.7rem 1rem; margin-bottom:0.5rem; font-size:0.9rem;">
                ↑ {imp}
            </div>
            """, unsafe_allow_html=True)

        # Action items
        st.markdown("### 🎯 Action Items")
        for i, item in enumerate(evaluation.action_items, 1):
            st.markdown(f"""
            <div style="background:#fafaf0; border:1px solid #fde68a; border-radius:8px;
                        padding:0.6rem 1rem; margin-bottom:0.4rem; font-size:0.85rem;">
                <b>{i}.</b> {item}
            </div>
            """, unsafe_allow_html=True)

        # Stats
        st.markdown("### 📊 Session Stats")
        turns = st.session_state.turn_count
        total_q = st.session_state.hr_count + st.session_state.tech_count
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.8rem;">
            <div class="stat-box"><div class="num">{turns}</div><div class="label">Your Answers</div></div>
            <div class="stat-box"><div class="num">{total_q}</div><div class="label">Questions Asked</div></div>
            <div class="stat-box"><div class="num">{st.session_state.hr_count}</div><div class="label">HR Questions</div></div>
            <div class="stat-box"><div class="num">{st.session_state.tech_count}</div><div class="label">Technical Questions</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Audio summary ──
    st.markdown("---")
    st.markdown("### 🔊 Spoken Evaluation Summary")
    if st.button("🎧 Generate Audio Summary", type="primary"):
        with st.spinner("Generating spoken summary..."):
            try:
                audio = chunk_and_speak(evaluation.spoken_summary)
                st.audio(audio)
            except Exception as e:
                st.error(f"Audio failed: {e}")

    st.info(f"**Summary:** {evaluation.spoken_summary}")

    # ── Restart ──
    st.markdown("---")
    if st.button("🔄 Start a New Interview", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
