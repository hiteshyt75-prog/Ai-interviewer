# utils/pipeline.py
"""
End-to-end analysis pipeline.
Orchestrates: audio → transcription → NLP → semantic → scoring → feedback → TTS.
"""

import os
from dataclasses import dataclass
from typing import Optional

from core.transcriber import transcribe_audio, TranscriptionResult
from core.nlp_analyzer import analyze_text, NLPAnalysis
from core.semantic_analyzer import compute_semantic_similarity, compute_keyword_coverage
from core.scorer import (
    score_hr_answer_with_text,
    score_technical_answer_with_text,
    InterviewScore
)
from core.feedback_generator import generate_feedback
from core.tts import chunk_and_speak
from config.roles import ROLES

ROLE_FALLBACK = {
    "mechanical_engineer": "general",
    "electrical_engineer": "general",
    "civil_engineer": "general",
    "chemical_engineer": "general",
    "software_engineer": "software_engineer",
    "data_scientist": "data_scientist",
    "devops_engineer": "devops_engineer",
    "product_manager": "product_manager",
}


@dataclass
class PipelineResult:
    transcription: TranscriptionResult
    nlp: NLPAnalysis
    score: InterviewScore
    feedback: dict
    audio_feedback_path: Optional[str]


def run_pipeline(
    audio_path: str,
    question: str,
    interview_type: str,
    role_key: str = "general",
    whisper_model: str = "base",
    generate_audio_feedback: bool = True
) -> PipelineResult:
    """
    Full analysis pipeline from audio file to feedback.
    """
    print(f"\n[Pipeline] Transcribing...")
    transcription = transcribe_audio(audio_path, model_size=whisper_model)

    print(f"[Pipeline] NLP analysis...")
    nlp = analyze_text(transcription.text)

    print(f"[Pipeline] Semantic relevance...")
    semantic = compute_semantic_similarity(
        candidate_answer=transcription.text,
        question=question,
        interview_type=interview_type
    )

    print(f"[Pipeline] Scoring...")
    roles_key = ROLE_FALLBACK.get(role_key, "general")

    if interview_type == "hr":
        score = score_hr_answer_with_text(
            question=question,
            text=transcription.text,
            nlp=nlp,
            semantic=semantic,
            wpm=transcription.words_per_minute
        )
    else:
        role_config = ROLES.get(roles_key, ROLES["general"])
        all_keywords = role_config["keywords"] + role_config["concepts"]
        kw_score, kw_matches = compute_keyword_coverage(transcription.text, all_keywords)
        score = score_technical_answer_with_text(
            question=question,
            role=role_key,
            text=transcription.text,
            nlp=nlp,
            semantic=semantic,
            keyword_matches=kw_matches,
            total_keywords=len(all_keywords),
            wpm=transcription.words_per_minute
        )

    print(f"[Pipeline] Generating feedback...")
    feedback = generate_feedback(score, nlp, transcription, question)

    audio_feedback_path = None
    if generate_audio_feedback:
        try:
            audio_feedback_path = chunk_and_speak(feedback["spoken_summary"])
        except Exception as e:
            print(f"[Pipeline] TTS failed (non-fatal): {e}")

    print(f"[Pipeline] Done. Score: {score.total_score}/100 ({score.grade})")

    return PipelineResult(
        transcription=transcription,
        nlp=nlp,
        score=score,
        feedback=feedback,
        audio_feedback_path=audio_feedback_path
    )
