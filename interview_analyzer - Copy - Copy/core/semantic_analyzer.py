# core/semantic_analyzer.py
"""
Semantic relevance analysis using sentence-transformers.
Computes cosine similarity between the candidate's answer
and ideal reference answers for each question type.
"""

from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Tuple
from dataclasses import dataclass


IDEAL_HR_ANSWERS = {
    "tell me about yourself": (
        "I am a motivated professional with strong technical skills and experience in my field. "
        "I have worked on meaningful projects, collaborated in teams, and consistently delivered results. "
        "I am looking to grow in a role where I can contribute meaningfully and continue learning."
    ),
    "strength": (
        "My key strength is my ability to solve complex problems systematically. "
        "I break challenges down into smaller components, research solutions thoroughly, "
        "and communicate clearly with stakeholders throughout the process."
    ),
    "weakness": (
        "I sometimes focus too much on detail when I should delegate. "
        "I have been actively working on this by setting clearer priorities, "
        "trusting team members with tasks, and reviewing outcomes rather than micromanaging."
    ),
    "conflict": (
        "I had a disagreement with a colleague about the project approach. "
        "I listened to their perspective, explained my reasoning with data, "
        "and we agreed on a compromise that led to a better outcome for the project."
    ),
    "achievement": (
        "I led a cross-functional team to deliver a product that increased revenue by 30 percent. "
        "I identified the bottleneck, restructured the workflow, and coordinated closely with engineering and design. "
        "The launch was ahead of schedule and received strong customer feedback."
    ),
    "default": (
        "I approach challenges systematically, communicate effectively with teams, "
        "take ownership of my responsibilities, and deliver measurable results. "
        "I learn quickly and adapt to new environments."
    )
}

IDEAL_TECHNICAL_ANSWERS = {
    "algorithm": (
        "An efficient algorithm should have optimal time and space complexity. "
        "I evaluate trade-offs between different approaches, consider edge cases, "
        "and choose the solution that balances performance with readability and maintainability."
    ),
    "system design": (
        "When designing a system I start by clarifying requirements and defining constraints. "
        "I then design the high-level architecture, choose appropriate databases and services, "
        "plan for scalability with load balancers and caching, and ensure fault tolerance."
    ),
    "debugging": (
        "I debug systematically by reproducing the issue, isolating the failing component, "
        "adding logging to trace state, forming a hypothesis, testing the fix, "
        "and writing a regression test to prevent recurrence."
    ),
    "default": (
        "I approach technical problems by first understanding requirements, analyzing constraints, "
        "designing a scalable and maintainable solution, implementing with clean code, "
        "testing thoroughly, and documenting the design decisions."
    )
}


@dataclass
class SemanticResult:
    similarity_score: float      # 0.0 – 1.0
    relevance_label: str         # "High", "Medium", "Low"
    matched_ideal: str           # Which ideal answer was matched


_model_cache = {}


def load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache a SentenceTransformer model."""
    if model_name not in _model_cache:
        print(f"[Semantic] Loading model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
        print(f"[Semantic] Model ready.")
    return _model_cache[model_name]


def _select_ideal(question: str, answer_bank: Dict[str, str]) -> Tuple[str, str]:
    """
    Select the best matching ideal answer from the bank
    by checking if any key phrase appears in the question.
    """
    q_lower = question.lower()
    for key, ideal in answer_bank.items():
        if key != "default" and key in q_lower:
            return key, ideal
    return "default", answer_bank["default"]


def compute_semantic_similarity(
    candidate_answer: str,
    question: str,
    interview_type: str = "hr",
    model_name: str = "all-MiniLM-L6-v2"
) -> SemanticResult:
    """
    Compute cosine similarity between the candidate's answer
    and the most relevant ideal reference answer.

    Args:
        candidate_answer: Transcribed text from the candidate.
        question: The interview question asked.
        interview_type: "hr" or "technical"
        model_name: SentenceTransformer model to use.

    Returns:
        SemanticResult with similarity score and label.
    """
    model = load_model(model_name)

    bank = IDEAL_HR_ANSWERS if interview_type == "hr" else IDEAL_TECHNICAL_ANSWERS
    matched_key, ideal_answer = _select_ideal(question, bank)

    candidate_emb = model.encode(candidate_answer, convert_to_tensor=True)
    ideal_emb = model.encode(ideal_answer, convert_to_tensor=True)

    score = float(util.cos_sim(candidate_emb, ideal_emb).item())
    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

    if score >= 0.60:
        label = "High"
    elif score >= 0.35:
        label = "Medium"
    else:
        label = "Low"

    return SemanticResult(
        similarity_score=round(score, 3),
        relevance_label=label,
        matched_ideal=matched_key
    )


def compute_keyword_coverage(text: str, keywords: List[str]) -> Tuple[float, List[str]]:
    """
    Compute what fraction of expected role keywords appear in the text.

    Returns:
        (coverage_score 0.0-1.0, list of matched keywords)
    """
    text_lower = text.lower()
    matched = [kw for kw in keywords if kw.lower() in text_lower]
    coverage = len(matched) / len(keywords) if keywords else 0.0
    return round(min(coverage * 2.5, 1.0), 3), matched  # scale: 40% hit = 1.0
