#!/usr/bin/env python3
# run_cli.py
"""
Command-line interface for the Interview Performance Analyzer.
Use this to test the system without the Streamlit UI.

Usage:
    python run_cli.py --audio path/to/answer.wav --question "Tell me about yourself." --type hr
    python run_cli.py --audio path/to/answer.wav --question "Explain the bias-variance tradeoff." --type technical --role data_scientist
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.pipeline import run_pipeline
from config.roles import ROLES


def main():
    parser = argparse.ArgumentParser(
        description="Interview Performance Analyzer — CLI",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--audio", required=True,
        help="Path to the audio file (.wav, .mp3, etc.)"
    )
    parser.add_argument(
        "--question", required=True,
        help="The interview question that was asked."
    )
    parser.add_argument(
        "--type", dest="interview_type", default="hr",
        choices=["hr", "technical"],
        help="Interview type: 'hr' or 'technical' (default: hr)"
    )
    parser.add_argument(
        "--role", default="general",
        choices=list(ROLES.keys()),
        help="Role key for technical interviews (default: general)"
    )
    parser.add_argument(
        "--model", default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip audio feedback generation"
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        sys.exit(1)

    print("\n" + "═" * 60)
    print("  INTERVIEW PERFORMANCE ANALYZER")
    print("═" * 60)
    print(f"  Question    : {args.question}")
    print(f"  Type        : {args.interview_type.upper()}")
    if args.interview_type == "technical":
        print(f"  Role        : {ROLES[args.role]['label']}")
    print(f"  Audio       : {args.audio}")
    print(f"  Whisper     : {args.model}")
    print("═" * 60 + "\n")

    result = run_pipeline(
        audio_path=args.audio,
        question=args.question,
        interview_type=args.interview_type,
        role_key=args.role,
        whisper_model=args.model,
        generate_audio_feedback=not args.no_audio
    )

    fb = result.feedback
    score = result.score
    tr = result.transcription
    nlp = result.nlp

    print("\n" + "═" * 60)
    print("  RESULTS")
    print("═" * 60)
    print(f"\n  SCORE : {score.total_score:.1f}/100   Grade: {score.grade}   [{fb['verdict']}]")
    print(f"\n  TRANSCRIPTION ({tr.word_count} words, {tr.words_per_minute} WPM):")
    print(f"  {tr.text[:300]}{'...' if len(tr.text) > 300 else ''}")

    print("\n  DIMENSION SCORES:")
    for dim in fb["dimension_feedback"]:
        bar = "█" * (dim["score_pct"] // 5) + "░" * (20 - dim["score_pct"] // 5)
        print(f"  {dim['name']:<30} [{bar}] {dim['score_pct']:3d}%")

    if fb["strengths"]:
        print("\n  STRENGTHS:")
        for s in fb["strengths"]:
            print(f"    ✓ {s}")

    if fb["improvements"]:
        print("\n  AREAS TO IMPROVE:")
        for imp in fb["improvements"]:
            print(f"    ✗ {imp}")

    print("\n  ADVICE:")
    print(f"  Pace   : {fb['pace_advice']}")
    print(f"  Filler : {fb['filler_advice']}")
    print(f"  Structure: {fb['structure_advice']}")

    print("\n  ACTION ITEMS:")
    for i, item in enumerate(fb["action_items"], 1):
        print(f"  {i}. {item}")

    if result.audio_feedback_path:
        print(f"\n  AUDIO FEEDBACK: {result.audio_feedback_path}")

    print("\n" + "═" * 60 + "\n")


if __name__ == "__main__":
    main()
