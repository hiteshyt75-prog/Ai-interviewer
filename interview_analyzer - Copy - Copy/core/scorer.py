# core/scorer.py
"""
Scoring engine for interview performance.
Combines NLP metrics, semantic similarity, and keyword coverage
into a weighted, explainable score per dimension.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from core.nlp_analyzer import NLPAnalysis
from core.semantic_analyzer import SemanticResult
from config.roles import SCORING_WEIGHTS, HR_COMPETENCIES


@dataclass
class DimensionScore:
    name: str
    raw_score: float        # 0.0 – 1.0
    weight: float           # contribution weight
    weighted_score: float   # raw * weight
    explanation: str        # human-readable reason


@dataclass
class InterviewScore:
    interview_type: str                    # "hr" or "technical"
    role: str
    question: str
    total_score: float                     # 0–100
    grade: str                             # A/B/C/D/F
    dimensions: List[DimensionScore]
    strengths: List[str]
    improvement_areas: List[str]
    keyword_matches: List[str]
    competency_scores: Dict[str, float]    # HR only


def _grade(score: float) -> str:
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def _fluency_score(nlp: NLPAnalysis, wpm: float) -> Tuple[float, str]:
    """
    Compute fluency from filler rate and speaking pace.
    Ideal WPM for interviews: 120–160.
    """
    # Filler penalty: 0 fillers/100w = 1.0, 10+ = 0.0
    filler_score = max(0.0, 1.0 - (nlp.filler_rate / 10.0))

    # Pace score: penalise extremes
    if 120 <= wpm <= 160:
        pace_score = 1.0
    elif 100 <= wpm < 120 or 160 < wpm <= 190:
        pace_score = 0.75
    elif wpm < 100 or wpm > 190:
        pace_score = 0.5
    else:
        pace_score = 0.5

    combined = (filler_score * 0.6) + (pace_score * 0.4)
    explanation = (
        f"Filler words: {nlp.filler_word_count} ({nlp.filler_rate:.1f}/100w); "
        f"Speaking pace: {wpm:.0f} WPM "
        f"({'ideal' if 120 <= wpm <= 160 else 'adjust pace'})"
    )
    return round(combined, 3), explanation


def _specificity_score(nlp: NLPAnalysis) -> Tuple[float, str]:
    """
    Specificity = quantified results + named entities + vocabulary richness.
    """
    quantity_score = 1.0 if nlp.has_quantified_results else 0.3
    entity_score = min(len(nlp.named_entities) / 5.0, 1.0)
    vocab_score = min(nlp.vocabulary_richness * 2.0, 1.0)

    combined = (quantity_score * 0.4) + (entity_score * 0.3) + (vocab_score * 0.3)
    explanation = (
        f"Quantified results: {'yes' if nlp.has_quantified_results else 'none found'}; "
        f"Named entities: {len(nlp.named_entities)}; "
        f"Vocabulary richness: {nlp.vocabulary_richness:.2f}"
    )
    return round(combined, 3), explanation


def _competency_scores(text: str) -> Dict[str, float]:
    """Score each HR competency by keyword presence density."""
    text_lower = text.lower()
    word_count = max(len(text_lower.split()), 1)
    scores = {}
    for comp, data in HR_COMPETENCIES.items():
        hits = sum(1 for kw in data["keywords"] if kw in text_lower)
        scores[comp] = round(min(hits / max(len(data["keywords"]) * 0.4, 1), 1.0), 3)
    return scores


def score_hr_answer(
    question: str,
    nlp: NLPAnalysis,
    semantic: SemanticResult,
    wpm: float
) -> InterviewScore:
    """
    Score an HR interview answer.
    """
    weights = SCORING_WEIGHTS["hr"]

    # 1. Relevance
    rel_score = semantic.similarity_score
    rel_dim = DimensionScore(
        name="Relevance",
        raw_score=rel_score,
        weight=weights["relevance"],
        weighted_score=rel_score * weights["relevance"],
        explanation=f"Semantic similarity to ideal answer: {rel_score:.2f} ({semantic.relevance_label})"
    )

    # 2. Structure (STAR + transitions)
    struct_score = nlp.structure_score
    star_parts = [k for k, v in nlp.star_coverage.items() if v]
    struct_dim = DimensionScore(
        name="Structure (STAR)",
        raw_score=struct_score,
        weight=weights["structure"],
        weighted_score=struct_score * weights["structure"],
        explanation=(
            f"STAR components detected: {', '.join(star_parts) if star_parts else 'none'}; "
            f"Transition words: {nlp.transition_word_count}"
        )
    )

    # 3. Competency coverage
    comp_scores = _competency_scores(nlp.filler_words_found.__class__.__name__)
    # Re-compute from text via a proxy
    comp_scores_raw = _competency_scores(" ".join(nlp.transition_words_found + nlp.filler_words_found))
    avg_comp = 0.5  # placeholder until text is passed — handled via wrapper below
    comp_dim = DimensionScore(
        name="Competency Coverage",
        raw_score=avg_comp,
        weight=weights["competency_coverage"],
        weighted_score=avg_comp * weights["competency_coverage"],
        explanation="Coverage of key HR competencies (teamwork, leadership, problem-solving, etc.)"
    )

    # 4. Specificity
    spec_score, spec_exp = _specificity_score(nlp)
    spec_dim = DimensionScore(
        name="Specificity",
        raw_score=spec_score,
        weight=weights["specificity"],
        weighted_score=spec_score * weights["specificity"],
        explanation=spec_exp
    )

    # 5. Fluency
    flu_score, flu_exp = _fluency_score(nlp, wpm)
    flu_dim = DimensionScore(
        name="Fluency",
        raw_score=flu_score,
        weight=weights["fluency"],
        weighted_score=flu_score * weights["fluency"],
        explanation=flu_exp
    )

    dimensions = [rel_dim, struct_dim, comp_dim, spec_dim, flu_dim]
    total = sum(d.weighted_score for d in dimensions) * 100

    strengths = []
    improvements = []
    for d in dimensions:
        if d.raw_score >= 0.7:
            strengths.append(f"{d.name}: {d.explanation}")
        elif d.raw_score < 0.5:
            improvements.append(f"{d.name}: {d.explanation}")

    return InterviewScore(
        interview_type="hr",
        role="hr",
        question=question,
        total_score=round(total, 1),
        grade=_grade(total),
        dimensions=dimensions,
        strengths=strengths,
        improvement_areas=improvements,
        keyword_matches=[],
        competency_scores={}
    )


def score_hr_answer_with_text(
    question: str,
    text: str,
    nlp: NLPAnalysis,
    semantic: SemanticResult,
    wpm: float
) -> InterviewScore:
    """
    Full HR scoring with access to raw text for competency analysis.
    """
    weights = SCORING_WEIGHTS["hr"]

    rel_score = semantic.similarity_score
    rel_dim = DimensionScore(
        name="Relevance",
        raw_score=rel_score,
        weight=weights["relevance"],
        weighted_score=rel_score * weights["relevance"],
        explanation=f"Semantic match to ideal answer: {rel_score:.2f} ({semantic.relevance_label})"
    )

    struct_score = nlp.structure_score
    star_parts = [k.title() for k, v in nlp.star_coverage.items() if v]
    struct_dim = DimensionScore(
        name="Structure (STAR)",
        raw_score=struct_score,
        weight=weights["structure"],
        weighted_score=struct_score * weights["structure"],
        explanation=(
            f"STAR components: {', '.join(star_parts) if star_parts else 'none detected'}; "
            f"Transition words used: {nlp.transition_word_count}"
        )
    )

    comp_scores = _competency_scores(text)
    avg_comp = sum(comp_scores.values()) / len(comp_scores) if comp_scores else 0.0
    comp_dim = DimensionScore(
        name="Competency Coverage",
        raw_score=avg_comp,
        weight=weights["competency_coverage"],
        weighted_score=avg_comp * weights["competency_coverage"],
        explanation=(
            "Top competencies: "
            + ", ".join(f"{k} ({v:.0%})" for k, v in sorted(comp_scores.items(), key=lambda x: -x[1])[:3])
        )
    )

    spec_score, spec_exp = _specificity_score(nlp)
    spec_dim = DimensionScore(
        name="Specificity & Examples",
        raw_score=spec_score,
        weight=weights["specificity"],
        weighted_score=spec_score * weights["specificity"],
        explanation=spec_exp
    )

    flu_score, flu_exp = _fluency_score(nlp, wpm)
    flu_dim = DimensionScore(
        name="Fluency & Delivery",
        raw_score=flu_score,
        weight=weights["fluency"],
        weighted_score=flu_score * weights["fluency"],
        explanation=flu_exp
    )

    dimensions = [rel_dim, struct_dim, comp_dim, spec_dim, flu_dim]
    total = sum(d.weighted_score for d in dimensions) * 100

    strengths = []
    improvements = []
    for d in dimensions:
        if d.raw_score >= 0.65:
            strengths.append(d.name)
        elif d.raw_score < 0.45:
            improvements.append(d.name)

    return InterviewScore(
        interview_type="hr",
        role="hr",
        question=question,
        total_score=round(total, 1),
        grade=_grade(total),
        dimensions=dimensions,
        strengths=strengths,
        improvement_areas=improvements,
        keyword_matches=[],
        competency_scores=comp_scores
    )


def score_technical_answer_with_text(
    question: str,
    role: str,
    text: str,
    nlp: NLPAnalysis,
    semantic: SemanticResult,
    keyword_matches: List[str],
    total_keywords: int,
    wpm: float
) -> InterviewScore:
    """
    Score a technical interview answer.
    """
    weights = SCORING_WEIGHTS["technical"]

    # 1. Relevance
    rel_score = semantic.similarity_score
    rel_dim = DimensionScore(
        name="Relevance",
        raw_score=rel_score,
        weight=weights["relevance"],
        weighted_score=rel_score * weights["relevance"],
        explanation=f"Semantic match to ideal technical answer: {rel_score:.2f} ({semantic.relevance_label})"
    )

    # 2. Keyword coverage
    kw_score = len(keyword_matches) / max(total_keywords * 0.4, 1)
    kw_score = min(kw_score, 1.0)
    kw_dim = DimensionScore(
        name="Technical Keyword Coverage",
        raw_score=kw_score,
        weight=weights["keyword_coverage"],
        weighted_score=kw_score * weights["keyword_coverage"],
        explanation=(
            f"Role keywords used: {len(keyword_matches)}/{total_keywords}. "
            f"Examples: {', '.join(keyword_matches[:5]) if keyword_matches else 'none'}"
        )
    )

    # 3. Depth (sentence length + vocabulary + specificity)
    depth_base = (
        min(nlp.avg_sentence_length / 20.0, 1.0) * 0.3 +
        min(nlp.vocabulary_richness * 2.5, 1.0) * 0.3 +
        (1.0 if nlp.has_quantified_results else 0.4) * 0.2 +
        min(nlp.sentence_count / 8.0, 1.0) * 0.2
    )
    depth_dim = DimensionScore(
        name="Technical Depth",
        raw_score=round(depth_base, 3),
        weight=weights["depth"],
        weighted_score=round(depth_base, 3) * weights["depth"],
        explanation=(
            f"Response length: {nlp.sentence_count} sentences; "
            f"Avg sentence length: {nlp.avg_sentence_length:.1f} words; "
            f"Vocabulary richness: {nlp.vocabulary_richness:.2f}"
        )
    )

    # 4. Structure
    struct_score = nlp.structure_score
    struct_dim = DimensionScore(
        name="Structure & Clarity",
        raw_score=struct_score,
        weight=weights["structure"],
        weighted_score=struct_score * weights["structure"],
        explanation=(
            f"Transition words: {nlp.transition_word_count}; "
            f"Organised reasoning detected: {'yes' if nlp.transition_word_count >= 2 else 'no'}"
        )
    )

    # 5. Fluency
    flu_score, flu_exp = _fluency_score(nlp, wpm)
    flu_dim = DimensionScore(
        name="Fluency & Delivery",
        raw_score=flu_score,
        weight=weights["fluency"],
        weighted_score=flu_score * weights["fluency"],
        explanation=flu_exp
    )

    dimensions = [rel_dim, kw_dim, depth_dim, struct_dim, flu_dim]
    total = sum(d.weighted_score for d in dimensions) * 100

    strengths = []
    improvements = []
    for d in dimensions:
        if d.raw_score >= 0.65:
            strengths.append(d.name)
        elif d.raw_score < 0.45:
            improvements.append(d.name)

    return InterviewScore(
        interview_type="technical",
        role=role,
        question=question,
        total_score=round(total, 1),
        grade=_grade(total),
        dimensions=dimensions,
        strengths=strengths,
        improvement_areas=improvements,
        keyword_matches=keyword_matches,
        competency_scores={}
    )
