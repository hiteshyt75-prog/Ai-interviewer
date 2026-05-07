# core/session.py
"""
Session manager for a live mock interview.
Tracks questions, answers, scores and session state across the full 20-question interview.
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from config.questions import HR_QUESTIONS, TECHNICAL_QUESTIONS, ENGINEERING_ROLES


@dataclass
class QuestionResult:
    index: int
    question: str
    interview_type: str        # "hr" or "technical"
    transcription_text: str
    words_per_minute: float
    total_score: float
    grade: str
    dimension_scores: List[dict]
    feedback: dict


@dataclass
class SessionConfig:
    role: str                  # key from ENGINEERING_ROLES
    hr_count: int              # number of HR questions
    technical_count: int       # number of technical questions
    whisper_model: str
    generate_audio: bool


@dataclass
class InterviewSession:
    config: SessionConfig
    questions: List[dict]      # list of {question, type} dicts
    results: List[QuestionResult] = field(default_factory=list)
    current_index: int = 0
    is_complete: bool = False

    @property
    def total_questions(self) -> int:
        return len(self.questions)

    @property
    def answered_count(self) -> int:
        return len(self.results)

    @property
    def current_question(self) -> Optional[dict]:
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    @property
    def progress_pct(self) -> float:
        return (self.answered_count / self.total_questions) * 100 if self.total_questions else 0

    @property
    def average_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.total_score for r in self.results) / len(self.results)

    @property
    def hr_average(self) -> float:
        hr = [r for r in self.results if r.interview_type == "hr"]
        return sum(r.total_score for r in hr) / len(hr) if hr else 0.0

    @property
    def technical_average(self) -> float:
        tech = [r for r in self.results if r.interview_type == "technical"]
        return sum(r.total_score for r in tech) / len(tech) if tech else 0.0


def build_session(config: SessionConfig) -> InterviewSession:
    """
    Build a shuffled list of questions for the session.
    Selects hr_count HR questions and technical_count technical questions,
    then interleaves them so the session alternates types naturally.
    """
    hr_pool = HR_QUESTIONS.copy()
    tech_pool = TECHNICAL_QUESTIONS.get(config.role, TECHNICAL_QUESTIONS["software_engineer"]).copy()

    random.shuffle(hr_pool)
    random.shuffle(tech_pool)

    hr_selected = hr_pool[:config.hr_count]
    tech_selected = tech_pool[:config.technical_count]

    # Interleave: start with HR, alternate where possible
    questions = []
    hr_iter = iter(hr_selected)
    tech_iter = iter(tech_selected)

    hr_done = False
    tech_done = False

    while not (hr_done and tech_done):
        if not hr_done:
            try:
                questions.append({"question": next(hr_iter), "type": "hr"})
            except StopIteration:
                hr_done = True
        if not tech_done:
            try:
                questions.append({"question": next(tech_iter), "type": "technical"})
            except StopIteration:
                tech_done = True

    return InterviewSession(config=config, questions=questions)


def compute_final_report(session: InterviewSession) -> dict:
    """
    Compute the full session report after all questions are answered.
    Returns a structured dict suitable for display and TTS.
    """
    results = session.results
    if not results:
        return {}

    total_avg = session.average_score
    hr_avg = session.hr_average
    tech_avg = session.technical_average

    # Grade
    def grade(s):
        if s >= 85: return "A"
        if s >= 70: return "B"
        if s >= 55: return "C"
        if s >= 40: return "D"
        return "F"

    # Best and worst questions
    sorted_results = sorted(results, key=lambda r: r.total_score)
    worst = sorted_results[:2]
    best = sorted_results[-2:]

    # Aggregate dimension scores across all questions
    dim_totals: Dict[str, List[float]] = {}
    for r in results:
        for d in r.dimension_scores:
            dim_totals.setdefault(d["name"], []).append(d["score_pct"])
    dim_averages = {k: round(sum(v) / len(v)) for k, v in dim_totals.items()}

    # Top improvement areas (lowest average dimensions)
    sorted_dims = sorted(dim_averages.items(), key=lambda x: x[1])
    top_improvements = [d[0] for d in sorted_dims[:3]]
    top_strengths = [d[0] for d in sorted_dims[-2:]]

    # Spoken summary for TTS
    verdict = (
        "Outstanding" if total_avg >= 85 else
        "Good" if total_avg >= 70 else
        "Average" if total_avg >= 55 else
        "Below average"
    )

    spoken = (
        f"Your mock interview is complete. "
        f"You answered {len(results)} questions across HR and technical rounds. "
        f"Your overall average score is {total_avg:.0f} out of 100, which is a grade {grade(total_avg)}. "
        f"Your performance was {verdict}. "
        f"Your HR average was {hr_avg:.0f} and your technical average was {tech_avg:.0f}. "
        f"Your strongest areas were {' and '.join(top_strengths)}. "
        f"Focus on improving {', '.join(top_improvements)} for your next session. "
        f"Well done for completing the full interview. Keep practising!"
    )

    return {
        "total_avg": round(total_avg, 1),
        "hr_avg": round(hr_avg, 1),
        "tech_avg": round(tech_avg, 1),
        "overall_grade": grade(total_avg),
        "verdict": verdict,
        "dim_averages": dim_averages,
        "top_strengths": top_strengths,
        "top_improvements": top_improvements,
        "best_questions": [{"q": r.question, "score": r.total_score} for r in best],
        "worst_questions": [{"q": r.question, "score": r.total_score} for r in worst],
        "per_question": [
            {
                "index": r.index + 1,
                "question": r.question,
                "type": r.interview_type,
                "score": r.total_score,
                "grade": r.grade,
            }
            for r in results
        ],
        "spoken_summary": spoken,
    }
