# core/feedback_generator.py
"""
Generates structured, natural-language feedback from a scored interview response.
All feedback is rule-based and grounded in actual metric values.
"""

from core.scorer import InterviewScore
from core.nlp_analyzer import NLPAnalysis
from core.transcriber import TranscriptionResult


def _pace_advice(wpm: float) -> str:
    if wpm < 100:
        return f"Your speaking pace was slow at {wpm:.0f} WPM. Aim for 120–160 WPM to sound more confident and engaged."
    elif wpm > 180:
        return f"Your speaking pace was fast at {wpm:.0f} WPM. Slowing down slightly to 120–160 WPM will improve clarity."
    else:
        return f"Your speaking pace of {wpm:.0f} WPM was well-controlled and easy to follow."


def _filler_advice(rate: float, words: list) -> str:
    if rate == 0:
        return "Excellent — no filler words detected."
    elif rate < 2:
        unique = ", ".join(set(words[:3]))
        return f"Low filler word usage ({rate:.1f} per 100 words). Watch for: {unique}."
    elif rate < 5:
        unique = ", ".join(set(words[:4]))
        return f"Moderate filler word usage ({rate:.1f} per 100 words): {unique}. Practice pausing silently instead."
    else:
        unique = ", ".join(set(words[:5]))
        return (
            f"High filler word usage ({rate:.1f} per 100 words): {unique}. "
            "Record yourself and replace fillers with deliberate pauses."
        )


def _star_advice(star_coverage: dict) -> str:
    missing = [k.title() for k, v in star_coverage.items() if not v]
    present = [k.title() for k, v in star_coverage.items() if v]
    if not missing:
        return "You covered all four STAR components (Situation, Task, Action, Result) — great structure."
    elif len(present) >= 2:
        return (
            f"You included {', '.join(present)} but missed {', '.join(missing)}. "
            "Complete the STAR method to make your answer more compelling."
        )
    else:
        return (
            "Your answer lacked clear STAR structure. "
            "Structure behavioural answers as: Situation → Task → Action → Result."
        )


def _quantification_advice(has_numbers: bool, numbers: list) -> str:
    if has_numbers:
        return f"Good use of specific metrics: {', '.join(numbers[:4])}. Quantified results are highly persuasive."
    else:
        return (
            "No specific metrics or numbers detected. "
            "Add quantified results (e.g. 'reduced time by 30%', 'managed a team of 5') to strengthen your answer."
        )


def _keyword_advice(matches: list, role: str) -> str:
    if not matches:
        return f"No role-specific keywords for '{role}' were detected. Use relevant technical terminology to signal expertise."
    elif len(matches) < 4:
        return f"Some relevant keywords found: {', '.join(matches)}. Incorporate more domain-specific terminology."
    else:
        return f"Strong keyword coverage: {', '.join(matches[:6])}{'...' if len(matches) > 6 else ''}. This demonstrates clear technical knowledge."


def generate_feedback(
    score: InterviewScore,
    nlp: NLPAnalysis,
    transcription: TranscriptionResult,
    question: str
) -> dict:
    """
    Generate structured feedback dict from analysis results.

    Returns a dict with keys:
        - summary: 2–3 sentence overall summary
        - dimension_feedback: per-dimension explanations
        - pace_advice: speaking pace guidance
        - filler_advice: filler word guidance
        - structure_advice: STAR / transition advice
        - specificity_advice: quantification and examples
        - keyword_advice: role keyword feedback (technical only)
        - spoken_summary: short paragraph suitable for TTS output
        - action_items: list of concrete improvement steps
    """
    total = score.total_score
    grade = score.grade
    interview_type = score.interview_type

    # Overall summary
    if total >= 85:
        verdict = "Outstanding"
        summary_tone = "You delivered a strong, well-structured answer that demonstrates clear competency."
    elif total >= 70:
        verdict = "Good"
        summary_tone = "You gave a solid answer with some areas that could be strengthened."
    elif total >= 55:
        verdict = "Average"
        summary_tone = "Your answer covered the basics but missed several opportunities to stand out."
    elif total >= 40:
        verdict = "Below Average"
        summary_tone = "Your answer needs significant improvement in structure, content, and delivery."
    else:
        verdict = "Needs Major Work"
        summary_tone = "Your answer requires substantial development across most dimensions."

    summary = (
        f"Overall performance: {verdict} (Score: {total:.1f}/100, Grade: {grade}). "
        f"{summary_tone} "
        f"Word count: {transcription.word_count}, Duration: {transcription.duration_seconds:.0f}s."
    )

    # Dimension feedback
    dimension_feedback = []
    for dim in score.dimensions:
        pct = int(dim.raw_score * 100)
        level = "Strong" if pct >= 70 else ("Developing" if pct >= 45 else "Weak")
        dimension_feedback.append({
            "name": dim.name,
            "score_pct": pct,
            "level": level,
            "weight_pct": int(dim.weight * 100),
            "explanation": dim.explanation
        })

    # Specific advice
    pace_advice = _pace_advice(transcription.words_per_minute)
    filler_advice = _filler_advice(nlp.filler_rate, nlp.filler_words_found)
    structure_advice = _star_advice(nlp.star_coverage) if interview_type == "hr" else (
        f"Used {nlp.transition_word_count} transition word(s). "
        + ("Good logical flow." if nlp.transition_word_count >= 2 else "Add transitions like 'first', 'therefore', 'as a result' to improve clarity.")
    )
    specificity_advice = _quantification_advice(nlp.has_quantified_results, nlp.numbers_mentioned)
    keyword_advice = _keyword_advice(score.keyword_matches, score.role) if interview_type == "technical" else ""

    # Action items
    action_items = []
    if nlp.filler_rate > 3:
        action_items.append("Practice speaking with deliberate pauses instead of filler words.")
    if not nlp.has_quantified_results:
        action_items.append("Add at least one specific metric or number to your answer.")
    if nlp.star_score < 0.75 and interview_type == "hr":
        action_items.append("Structure your answer using the STAR method: Situation, Task, Action, Result.")
    if transcription.words_per_minute < 110:
        action_items.append("Speed up your delivery slightly — aim for 120–160 words per minute.")
    if transcription.words_per_minute > 180:
        action_items.append("Slow down your delivery — pause between ideas to allow the interviewer to absorb your points.")
    if nlp.transition_word_count < 2:
        action_items.append("Use transition words (firstly, therefore, as a result) to structure your thinking.")
    if interview_type == "technical" and len(score.keyword_matches) < 3:
        action_items.append(f"Use more role-specific terminology relevant to the {score.role} position.")
    if not action_items:
        action_items.append("Continue practicing and refining your delivery for consistency.")

    # Spoken summary (for TTS — concise and natural)
    spoken = (
        f"Here is your feedback. "
        f"You scored {total:.0f} out of 100, which is a grade {grade}. "
        f"{summary_tone} "
        f"{pace_advice} "
        f"{filler_advice} "
        f"{'Your main improvement areas are: ' + ', '.join(score.improvement_areas[:2]) + '.' if score.improvement_areas else 'Keep up the strong performance.'}"
    )

    return {
        "summary": summary,
        "verdict": verdict,
        "dimension_feedback": dimension_feedback,
        "pace_advice": pace_advice,
        "filler_advice": filler_advice,
        "structure_advice": structure_advice,
        "specificity_advice": specificity_advice,
        "keyword_advice": keyword_advice,
        "spoken_summary": spoken,
        "action_items": action_items,
        "strengths": score.strengths,
        "improvements": score.improvement_areas
    }
