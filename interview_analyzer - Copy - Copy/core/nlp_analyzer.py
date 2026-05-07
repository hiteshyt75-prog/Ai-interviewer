# core/nlp_analyzer.py
"""
NLP analysis using spaCy.
Extracts linguistic features: sentence structure, named entities,
filler words, fluency metrics, and STAR method detection.
"""

import spacy
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


FILLER_WORDS = {
    "um", "uh", "like", "you know", "basically", "literally",
    "actually", "honestly", "right", "so", "well", "kind of",
    "sort of", "i mean", "yeah", "okay"
}

STAR_KEYWORDS = {
    "situation": ["situation", "context", "background", "when", "at", "during", "working at"],
    "task": ["task", "responsibility", "role", "objective", "goal", "needed to", "had to", "was asked"],
    "action": ["action", "did", "implemented", "created", "built", "developed", "approached", "decided", "led", "managed"],
    "result": ["result", "outcome", "achieved", "improved", "increased", "reduced", "saved", "impact", "effect", "success"]
}

TRANSITION_WORDS = {
    "firstly", "secondly", "finally", "however", "therefore",
    "consequently", "moreover", "furthermore", "in addition",
    "as a result", "on the other hand", "for example", "for instance",
    "in conclusion", "to summarize", "additionally"
}


@dataclass
class NLPAnalysis:
    sentence_count: int
    avg_sentence_length: float
    filler_word_count: int
    filler_words_found: List[str]
    filler_rate: float                   # fillers per 100 words
    named_entities: List[Tuple[str, str]]
    transition_word_count: int
    transition_words_found: List[str]
    star_coverage: Dict[str, bool]
    star_score: float                    # 0.0 – 1.0
    vocabulary_richness: float           # unique / total words
    passive_voice_count: int
    has_quantified_results: bool
    numbers_mentioned: List[str]
    structure_score: float               # 0.0 – 1.0


_nlp_model = None


def load_nlp() -> spacy.Language:
    """Load and cache the spaCy model."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            print("[NLP] spaCy model 'en_core_web_sm' loaded.")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _nlp_model


def _detect_filler_words(text: str) -> Tuple[int, List[str]]:
    """Count filler words/phrases in text (case-insensitive)."""
    lower = text.lower()
    found = []
    count = 0
    for filler in FILLER_WORDS:
        occurrences = len(re.findall(r'\b' + re.escape(filler) + r'\b', lower))
        if occurrences > 0:
            found.append(filler)
            count += occurrences
    return count, found


def _detect_star_method(text: str) -> Dict[str, bool]:
    """Detect whether STAR method components are present."""
    lower = text.lower()
    coverage = {}
    for component, keywords in STAR_KEYWORDS.items():
        coverage[component] = any(kw in lower for kw in keywords)
    return coverage


def _detect_transition_words(text: str) -> Tuple[int, List[str]]:
    """Detect structured transition words."""
    lower = text.lower()
    found = []
    count = 0
    for tw in TRANSITION_WORDS:
        if tw in lower:
            found.append(tw)
            count += 1
    return count, found


def _detect_passive_voice(doc) -> int:
    """Count passive voice constructions using dependency tags."""
    count = 0
    for token in doc:
        if token.dep_ == "nsubjpass":
            count += 1
    return count


def _extract_numbers(text: str) -> List[str]:
    """Extract numeric values and percentages (e.g. '30%', '$1M', '3 months')."""
    pattern = r'\b\d+(?:\.\d+)?(?:%|k|m|b|x|million|billion|thousand|percent)?\b'
    return re.findall(pattern, text.lower())


def analyze_text(text: str) -> NLPAnalysis:
    """
    Full NLP analysis of a transcribed interview response.

    Args:
        text: Raw transcription string.

    Returns:
        NLPAnalysis dataclass with all computed metrics.
    """
    nlp = load_nlp()
    doc = nlp(text)

    sentences = list(doc.sents)
    sentence_count = len(sentences)
    avg_sentence_length = (
        sum(len(list(s)) for s in sentences) / sentence_count
        if sentence_count > 0 else 0.0
    )

    words = [t.text.lower() for t in doc if t.is_alpha]
    total_words = len(words)
    unique_words = len(set(words))
    vocab_richness = unique_words / total_words if total_words > 0 else 0.0

    filler_count, fillers_found = _detect_filler_words(text)
    filler_rate = (filler_count / total_words * 100) if total_words > 0 else 0.0

    transition_count, transitions_found = _detect_transition_words(text)
    star_coverage = _detect_star_method(text)
    star_score = sum(star_coverage.values()) / len(star_coverage)

    entities = [(ent.text, ent.label_) for ent in doc.ents]
    passive_count = _detect_passive_voice(doc)
    numbers = _extract_numbers(text)
    has_quantified = len(numbers) > 0

    # Structure score: based on STAR coverage + transition words
    structure_score = min(1.0, (star_score * 0.6) + (min(transition_count, 5) / 5 * 0.4))

    return NLPAnalysis(
        sentence_count=sentence_count,
        avg_sentence_length=round(avg_sentence_length, 1),
        filler_word_count=filler_count,
        filler_words_found=fillers_found,
        filler_rate=round(filler_rate, 2),
        named_entities=entities,
        transition_word_count=transition_count,
        transition_words_found=transitions_found,
        star_coverage=star_coverage,
        star_score=round(star_score, 2),
        vocabulary_richness=round(vocab_richness, 3),
        passive_voice_count=passive_count,
        has_quantified_results=has_quantified,
        numbers_mentioned=numbers,
        structure_score=round(structure_score, 2)
    )
