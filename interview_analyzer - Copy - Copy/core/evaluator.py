# core/evaluator.py
"""
End-of-session evaluation using GPT-4.
Analyses the full conversation transcript and produces:
  - Overall score (0-100)
  - Per-dimension scores with explanations
  - Strengths and improvement areas
  - Actionable recommendations
  - A spoken summary for TTS
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict
from openai import OpenAI
from core.interviewer import Turn


@dataclass
class DimensionEval:
    name: str
    score: int           # 0-100
    explanation: str
    examples: List[str]  # direct quotes or paraphrases from answers


@dataclass
class SessionEvaluation:
    overall_score: int
    overall_grade: str
    verdict: str
    dimensions: List[DimensionEval]
    top_strengths: List[str]
    top_improvements: List[str]
    action_items: List[str]
    best_answer_summary: str
    weakest_answer_summary: str
    spoken_summary: str
    raw_json: dict


EVAL_DIMENSIONS = [
    "Communication Clarity",
    "Technical Knowledge Depth",
    "Structured Thinking",
    "Use of Concrete Examples",
    "Confidence and Delivery",
    "Problem Solving Approach",
    "Self Awareness",
]


def _grade(score: int) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def _verdict(score: int) -> str:
    if score >= 85: return "Outstanding"
    if score >= 70: return "Good"
    if score >= 55: return "Average"
    if score >= 40: return "Below Average"
    return "Needs Significant Work"


def _format_transcript(turns: List[Turn]) -> str:
    """Format conversation history as readable transcript."""
    lines = []
    for turn in turns:
        label = "INTERVIEWER" if turn.role == "interviewer" else "CANDIDATE"
        lines.append(f"{label}: {turn.text}")
    return "\n\n".join(lines)


def evaluate_session(
    turns: List[Turn],
    role: str,
    api_key: str
) -> SessionEvaluation:
    """
    Send the full transcript to GPT-4 for evaluation.
    Returns a structured SessionEvaluation object.
    """
    client = OpenAI(api_key=api_key)
    transcript = _format_transcript(turns)

    system_prompt = """You are an expert interview coach and evaluator.
You will receive a full interview transcript between an AI interviewer and a candidate.
Your job is to evaluate the candidate's performance honestly and helpfully.

You must respond with ONLY a valid JSON object — no markdown, no preamble, no explanation outside the JSON.

The JSON must have exactly this structure:
{
  "overall_score": <integer 0-100>,
  "dimensions": [
    {
      "name": "<dimension name>",
      "score": <integer 0-100>,
      "explanation": "<2-3 sentence explanation of this score>",
      "examples": ["<paraphrase or short quote from the candidate's actual answer>"]
    }
  ],
  "top_strengths": ["<strength 1>", "<strength 2>", "<strength 3>"],
  "top_improvements": ["<area 1>", "<area 2>", "<area 3>"],
  "action_items": [
    "<specific, actionable recommendation 1>",
    "<specific, actionable recommendation 2>",
    "<specific, actionable recommendation 3>",
    "<specific, actionable recommendation 4>"
  ],
  "best_answer_summary": "<1-2 sentences describing the candidate's strongest answer>",
  "weakest_answer_summary": "<1-2 sentences describing the candidate's weakest answer>",
  "spoken_summary": "<A 4-6 sentence spoken summary suitable for text-to-speech. Natural, conversational, encouraging but honest. Mention the score, grade, top strength, and top area to improve.>"
}"""

    user_prompt = f"""Please evaluate this candidate for a {role.replace('_', ' ')} position.

Evaluate across these dimensions:
{chr(10).join(f'- {d}' for d in EVAL_DIMENSIONS)}

INTERVIEW TRANSCRIPT:
{transcript}

Remember: respond with ONLY the JSON object."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    raw_text = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    data = json.loads(raw_text)

    overall = data.get("overall_score", 50)
    dims = []
    for d in data.get("dimensions", []):
        dims.append(DimensionEval(
            name=d.get("name", ""),
            score=d.get("score", 50),
            explanation=d.get("explanation", ""),
            examples=d.get("examples", [])
        ))

    return SessionEvaluation(
        overall_score=overall,
        overall_grade=_grade(overall),
        verdict=_verdict(overall),
        dimensions=dims,
        top_strengths=data.get("top_strengths", []),
        top_improvements=data.get("top_improvements", []),
        action_items=data.get("action_items", []),
        best_answer_summary=data.get("best_answer_summary", ""),
        weakest_answer_summary=data.get("weakest_answer_summary", ""),
        spoken_summary=data.get("spoken_summary", ""),
        raw_json=data
    )
