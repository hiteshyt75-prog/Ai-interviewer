# app.py
"""
Live AI Mock Interview — Voice Chatbot
GPT-4o-mini (conversation + TTS echo voice) + Whisper (speech-to-text)
"""

import streamlit as st
import sys
import os
import tempfile
import base64
import hashlib
from typing import Optional
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.transcriber import transcribe_audio
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
    box-shadow: 0 4px 20px rgba(99,102,241,0.25);
}
.chat-bubble-ai .speaker {
    font-size: 0.72rem; color: #a5b4fc; margin-bottom: 0.4rem;
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}
.chat-bubble-ai .msg { font-size: 1rem; line-height: 1.6; }
.followup-tag {
    display: inline-block; background: #7c3aed; font-size: 0.65rem;
    padding: 2px 8px; border-radius: 99px; margin-left: 8px; vertical-align: middle;
}
.chat-bubble-user {
    background: #f1f5f9;
    border-radius: 18px 18px 4px 18px;
    padding: 1.2rem 1.5rem;
    color: #1e293b;
    margin: 0.8rem 0 0.8rem auto;
    max-width: 85%;
    border: 1px solid #e2e8f0;
}
.chat-bubble-user .speaker {
    font-size: 0.72rem; color: #64748b; margin-bottom: 0.4rem;
    font-weight: 600; letter-spacing: 1px; text-transform: uppercase;
}
.chat-bubble-user .msg { font-size: 1rem; line-height: 1.6; }

.score-hero {
    border-radius: 20px; padding: 2.5rem;
    color: white; text-align: center; margin-bottom: 2rem;
}
.dim-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-left: 4px solid #6366f1; border-radius: 10px;
    padding: 1rem 1.2rem; margin-bottom: 0.75rem;
}
.dim-card.strong { border-left-color: #10b981; }
.dim-card.weak   { border-left-color: #ef4444; }
.stat-box {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 1.2rem; text-align: center;
}
.stat-box .num   { font-size: 2rem; font-weight: 700; color: #6366f1; }
.stat-box .label { font-size: 0.8rem; color: #64748b; margin-top: 0.2rem; }
.stButton > button { border-radius: 10px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def autoplay_audio(file_path: str):
    """Embed audio with autoplay — plays immediately without user interaction."""
    if not file_path or not os.path.exists(file_path):
        return
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<audio autoplay style="display:none">'
        f'<source src="data:audio/mp3;base64,{b64}" type="audio/mp3">'
        f'</audio>',
        unsafe_allow_html=True
    )


def speak_openai(text: str, voice: str = "echo") -> Optional[str]:
    """Generate speech via OpenAI TTS. Returns MP3 file path or None."""
    if not st.session_state.get("generate_audio", True):
        return None
    try:
        client = OpenAI(api_key=st.session_state.api_key)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text
        )
        response.stream_to_file(tmp.name)
        return tmp.name
    except Exception as e:
        print(f"[TTS] Failed: {e}")
        return None


def audio_hash(audio_bytes: bytes) -> str:
    """MD5 hash of audio bytes — used to detect duplicate submissions."""
    return hashlib.md5(audio_bytes).hexdigest()


def show_nav(warn: bool = False):
    """Home button — always top-left."""
    col_home, col_warn = st.columns([1, 5])
    with col_home:
        if st.button("🏠 Home", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
    if warn:
        with col_warn:
            st.caption("⚠️ Going home will end the interview and lose all progress.")


def render_chat(autoplay_last: bool = False):
    """Render all chat bubbles. Autoplay only the most recent AI message."""
    history = st.session_state.chat_history_display
    for i, msg in enumerate(history):
        is_last = (i == len(history) - 1)
        if msg["role"] == "ai":
            tag = '<span class="followup-tag">FOLLOW-UP</span>' if msg.get("is_followup") else ""
            st.markdown(f"""
            <div class="chat-bubble-ai">
                <div class="speaker">🤖 AI INTERVIEWER {tag}</div>
                <div class="msg">{msg["text"]}</div>
            </div>""", unsafe_allow_html=True)
            if autoplay_last and is_last and msg.get("audio_path"):
                autoplay_audio(msg["audio_path"])
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;">
            <div class="chat-bubble-user">
                <div class="speaker">🎤 YOU</div>
                <div class="msg">{msg["text"]}</div>
            </div></div>""", unsafe_allow_html=True)


# ─── Session state ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "page": "setup",
        "interviewer": None,
        "chat_history_display": [],
        "evaluation": None,
        "turn_count": 0,
        "whisper_model": "base",
        "api_key": "",
        "role": "software_engineer",
        "hr_count": 5,
        "tech_count": 10,
        "generate_audio": True,
        "autoplay_pending": False,
        # Key fix: track the hash of the last processed audio
        # so we never process the same recording twice
        "last_audio_hash": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SETUP
# ═══════════════════════════════════════════════════════════════════════════════

if st.session_state.page == "setup":

    st.markdown("""
    <div class="hero">
        <h1>🤖 AI Mock Interviewer</h1>
        <p>Conversational · Voice-to-Voice · GPT-4o-mini · Echo Voice</p>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔑 OpenAI API Key")
        api_key = st.text_input(
            "API Key", type="password", placeholder="sk-...",
            label_visibility="collapsed"
        )
        st.caption("Get yours at [platform.openai.com](https://platform.openai.com/api-keys) — never stored.")

        st.markdown("### 🎓 Your Role")
        role = st.selectbox(
            "Role", options=list(ENGINEERING_ROLES.keys()),
            format_func=lambda k: ENGINEERING_ROLES[k],
            label_visibility="collapsed"
        )

        generate_audio = st.checkbox("🔊 AI speaks using Echo voice (autoplays)", value=True)

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
        1. GPT-4o-mini greets you with a unique opening question — **plays in Echo voice automatically**
        2. **Record your answer** using the mic below
        3. Stop recording — it submits automatically
        4. GPT reads your answer → asks a natural follow-up or moves on
        5. AI response **plays automatically**
        6. Repeat → **Full GPT-4o-mini evaluation at the end**
        """)

    st.markdown("---")
    start_col, _ = st.columns([1, 2])
    with start_col:
        ready = bool(api_key) and total > 0
        if not api_key:
            st.warning("Enter your OpenAI API key to begin.")
        if st.button("🚀 Start Interview", type="primary",
                     use_container_width=True, disabled=not ready):
            with st.spinner("Preparing your interviewer..."):
                try:
                    interviewer = AIInterviewer(
                        api_key=api_key,
                        role=role,
                        hr_count=hr_count,
                        tech_count=tech_count
                    )
                    opening = interviewer.start_interview()

                    st.session_state.api_key = api_key
                    st.session_state.generate_audio = generate_audio

                    audio_path = speak_openai(opening) if generate_audio else None

                    st.session_state.interviewer = interviewer
                    st.session_state.role = role
                    st.session_state.hr_count = hr_count
                    st.session_state.tech_count = tech_count
                    st.session_state.chat_history_display = [{
                        "role": "ai",
                        "text": opening,
                        "is_followup": False,
                        "audio_path": audio_path
                    }]
                    st.session_state.turn_count = 0
                    st.session_state.autoplay_pending = True
                    st.session_state.last_audio_hash = None
                    st.session_state.page = "interview"
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: INTERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "interview":

    interviewer: AIInterviewer = st.session_state.interviewer
    total_q = interviewer.state.hr_count + interviewer.state.tech_count
    asked = interviewer.state.questions_asked

    show_nav(warn=True)

    st.markdown("""
    <div class="hero" style="padding:1.2rem 2rem;">
        <h1 style="font-size:1.8rem;">🤖 Interview in Progress</h1>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Questions Covered", f"{asked}/{total_q}")
    with c2: st.metric("Your Answers", st.session_state.turn_count)
    with c3: st.metric("Role", ENGINEERING_ROLES.get(st.session_state.role, ""))
    st.progress(min(asked / max(total_q, 1), 1.0))
    st.markdown("---")

    # Render chat — autoplay only when a new AI message just arrived
    render_chat(autoplay_last=st.session_state.autoplay_pending)
    st.session_state.autoplay_pending = False

    # Interview complete
    if interviewer.is_complete:
        st.markdown("---")
        st.success("✅ Interview complete! Ready to see your evaluation?")
        if st.button("📊 Generate My Report", type="primary"):
            with st.spinner("GPT-4o-mini is evaluating your full session..."):
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

    # ── Recording ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎤 Your Answer")
    st.caption("Click the mic to record. Click again to stop — submits automatically.")

    audio_input = st.audio_input("🎙️ Record your answer")

    if audio_input is not None:
        # ── Duplicate detection ──
        # Read bytes and hash them. If this hash matches the last one
        # we already processed, it's a Streamlit rerun — skip it.
        raw_bytes = audio_input.read()
        current_hash = audio_hash(raw_bytes)

        if current_hash == st.session_state.last_audio_hash:
            # Same audio as last submission — do nothing, wait for new recording
            st.info("✅ Answer submitted. Record your next answer when ready.")
            st.stop()

        # New audio — process it
        st.session_state.last_audio_hash = current_hash

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(raw_bytes)
        tmp.close()

        with st.spinner("Transcribing your answer..."):
            try:
                transcription = transcribe_audio(
                    tmp.name,
                    api_key=st.session_state.api_key
                )
                candidate_text = transcription.text.strip()
            except Exception as e:
                st.error(f"Transcription failed: {e}")
                st.stop()

        if not candidate_text or len(candidate_text.split()) < 2:
            st.warning("Answer was too short or silent — please try again.")
            st.session_state.last_audio_hash = None  # allow retry
            st.stop()

        # Add user bubble
        st.session_state.chat_history_display.append({
            "role": "user",
            "text": candidate_text,
            "audio_path": None
        })
        st.session_state.turn_count += 1

        # GPT-4o-mini response
        with st.spinner("GPT-4o-mini is thinking..."):
            try:
                ai_response = interviewer.respond_to_answer(candidate_text)
                is_followup = (
                    len(interviewer.state.turns) >= 2 and
                    interviewer.state.turns[-1].is_followup
                )
                ai_audio = speak_openai(ai_response) if st.session_state.generate_audio else None
            except Exception as e:
                st.error(f"AI response failed: {e}")
                st.stop()

        # Add AI bubble
        st.session_state.chat_history_display.append({
            "role": "ai",
            "text": ai_response,
            "is_followup": is_followup,
            "audio_path": ai_audio
        })

        st.session_state.autoplay_pending = True
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: REPORT
# ═══════════════════════════════════════════════════════════════════════════════

elif st.session_state.page == "report":
    from core.evaluator import SessionEvaluation
    evaluation: SessionEvaluation = st.session_state.evaluation

    show_nav(warn=False)

    grade_colors = {
        "A": "#059669", "B": "#2563eb",
        "C": "#d97706", "D": "#dc2626", "F": "#7c3aed"
    }
    gc = grade_colors.get(evaluation.overall_grade, "#4f46e5")

    st.markdown("""
    <div class="hero">
        <h1>📋 Evaluation Report</h1>
        <p>GPT-4o-mini full session analysis</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="score-hero" style="background:linear-gradient(135deg,{gc},{gc}99);">
        <div style="font-size:0.9rem;opacity:0.85;margin-bottom:0.5rem;
                    text-transform:uppercase;letter-spacing:2px;">
            Overall · {ENGINEERING_ROLES.get(st.session_state.role,'')}
        </div>
        <div style="font-size:4.5rem;font-weight:800;line-height:1;">
            {evaluation.overall_score}<span style="font-size:2rem;">/100</span>
        </div>
        <div style="font-size:1.6rem;margin-top:0.3rem;font-weight:600;">
            Grade {evaluation.overall_grade} · {evaluation.verdict}
        </div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("### 📐 Dimension Scores")
        for dim in evaluation.dimensions:
            bar_color = "#10b981" if dim.score >= 70 else ("#f59e0b" if dim.score >= 50 else "#ef4444")
            card_cls = "strong" if dim.score >= 70 else ("weak" if dim.score < 50 else "")
            eg_html = (
                f'<div style="font-size:0.78rem;color:#6b7280;margin-top:0.4rem;font-style:italic;">'
                f'e.g. "{dim.examples[0]}"</div>'
            ) if dim.examples else ""
            st.markdown(f"""
            <div class="dim-card {card_cls}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-weight:600;font-size:0.95rem;">{dim.name}</span>
                    <span style="font-weight:800;color:{bar_color};font-size:1.1rem;">{dim.score}/100</span>
                </div>
                <div style="font-size:0.85rem;color:#4b5563;margin-top:0.3rem;line-height:1.5;">
                    {dim.explanation}</div>
                {eg_html}
            </div>""", unsafe_allow_html=True)
            st.progress(dim.score / 100)

        if evaluation.best_answer_summary:
            st.success(f"**Best answer:** {evaluation.best_answer_summary}")
        if evaluation.weakest_answer_summary:
            st.warning(f"**Most room to improve:** {evaluation.weakest_answer_summary}")

        with st.expander("📜 Full Interview Transcript"):
            for msg in st.session_state.chat_history_display:
                label = "🤖 Interviewer" if msg["role"] == "ai" else "🎤 You"
                st.markdown(f"**{label}:** {msg['text']}")
                st.markdown("---")

    with right:
        st.markdown("### ✅ Top Strengths")
        for s in evaluation.top_strengths:
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #86efac;border-radius:8px;
                        padding:0.7rem 1rem;margin-bottom:0.5rem;font-size:0.9rem;">✓ {s}</div>
            """, unsafe_allow_html=True)

        st.markdown("### ⚡ Areas to Improve")
        for imp in evaluation.top_improvements:
            st.markdown(f"""
            <div style="background:#fff7ed;border:1px solid #fdba74;border-radius:8px;
                        padding:0.7rem 1rem;margin-bottom:0.5rem;font-size:0.9rem;">↑ {imp}</div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Action Items")
        for i, item in enumerate(evaluation.action_items, 1):
            st.markdown(f"""
            <div style="background:#fafaf0;border:1px solid #fde68a;border-radius:8px;
                        padding:0.6rem 1rem;margin-bottom:0.4rem;font-size:0.85rem;">
                <b>{i}.</b> {item}</div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Session Stats")
        turns = st.session_state.turn_count
        total_q = st.session_state.hr_count + st.session_state.tech_count
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;">
            <div class="stat-box"><div class="num">{turns}</div><div class="label">Your Answers</div></div>
            <div class="stat-box"><div class="num">{total_q}</div><div class="label">Questions Asked</div></div>
            <div class="stat-box"><div class="num">{st.session_state.hr_count}</div><div class="label">HR Questions</div></div>
            <div class="stat-box"><div class="num">{st.session_state.tech_count}</div><div class="label">Technical Questions</div></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔊 Spoken Summary")
    if st.button("🎧 Play Audio Summary", type="primary"):
        with st.spinner("Generating Echo voice summary..."):
            try:
                audio = speak_openai(evaluation.spoken_summary)
                if audio:
                    autoplay_audio(audio)
                    st.success("Playing now...")
            except Exception as e:
                st.error(f"Audio failed: {e}")

    st.info(f"**Summary:** {evaluation.spoken_summary}")

    st.markdown("---")
    if st.button("🔄 Start New Interview", type="primary"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()
