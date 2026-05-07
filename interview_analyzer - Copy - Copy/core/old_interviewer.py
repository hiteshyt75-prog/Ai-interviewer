# core/interviewer.py
"""
AI Interviewer powered by OpenAI GPT-4.
Manages the full conversational interview loop:
  - Asks opening question
  - Listens to candidate's answer
  - Decides: follow-up, probe deeper, or move to next topic
  - Maintains conversation history
  - Produces end-of-session evaluation
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional
from openai import OpenAI
from config.questions import (
    SEED_QUESTIONS, HR_SEED_QUESTIONS,
    OPENING_QUESTIONS, ENGINEERING_ROLES
)


@dataclass
class Turn:
    role: str        # "interviewer" or "candidate"
    text: str
    is_question: bool = False
    is_followup: bool = False


@dataclass
class InterviewerState:
    role: str
    hr_count: int
    tech_count: int
    turns: List[Turn] = field(default_factory=list)
    questions_asked: int = 0
    hr_asked: int = 0
    tech_asked: int = 0
    used_seed_questions: List[str] = field(default_factory=list)
    consecutive_followups: int = 0   # limit follow-ups per topic to 2
    is_complete: bool = False
    current_topic: str = ""


def _build_system_prompt(role: str, hr_count: int, tech_count: int) -> str:
    role_label = ENGINEERING_ROLES.get(role, role.replace("_", " ").title())
    total = hr_count + tech_count
    return f"""You are a professional, senior interviewer conducting a mock job interview for a {role_label} position.

INTERVIEW STRUCTURE:
- Total questions: {total} (approximately {hr_count} HR/behavioural + {tech_count} technical)
- You are currently conducting the interview turn by turn.

YOUR BEHAVIOUR:
1. Ask ONE question or follow-up at a time. Never ask two questions in one turn.
2. After the candidate answers, decide ONE of:
   a) Ask a follow-up if the answer was vague, incomplete, or interesting enough to probe (max 2 follow-ups per topic)
   b) Give a brief 1-sentence acknowledgement ("Interesting, thank you.") then move to a new question
   c) If the candidate clearly struggled, be encouraging but move on
3. Follow-ups should be natural and directly reference what the candidate just said.
4. Balance between HR and technical questions as instructed.
5. Be professional, warm but firm. Do not give hints or reveal scoring.
6. Keep your responses concise — you are an interviewer, not a teacher.
7. When you have covered enough questions, say exactly: "That brings us to the end of our interview. Thank you for your time." and nothing else.

RESPONSE FORMAT:
- Respond with ONLY what you would say out loud as the interviewer.
- No stage directions, no labels, no markdown.
- Sound natural and human.
"""


def _build_messages(state: InterviewerState, system_prompt: str) -> list:
    """Convert turn history to OpenAI message format."""
    messages = [{"role": "system", "content": system_prompt}]
    for turn in state.turns:
        role = "assistant" if turn.role == "interviewer" else "user"
        messages.append({"role": role, "content": turn.text})
    return messages


def _get_next_seed_question(state: InterviewerState) -> str:
    """Pick the next unused seed question based on remaining HR/tech balance."""
    need_hr = state.hr_asked < state.hr_count
    need_tech = state.tech_asked < state.tech_count

    # Decide which type to ask next
    if need_hr and need_tech:
        # Alternate, slightly favour tech for engineers
        ask_hr = (state.hr_asked <= state.tech_asked * 0.6)
    elif need_hr:
        ask_hr = True
    else:
        ask_hr = False

    if ask_hr:
        pool = [q for q in HR_SEED_QUESTIONS if q not in state.used_seed_questions]
        if not pool:
            pool = HR_SEED_QUESTIONS
    else:
        pool = [q for q in SEED_QUESTIONS.get(state.role, []) if q not in state.used_seed_questions]
        if not pool:
            pool = SEED_QUESTIONS.get(state.role, ["Tell me about a technical challenge you solved recently."])

    chosen = random.choice(pool)
    state.used_seed_questions.append(chosen)
    return chosen


class AIInterviewer:
    """
    Conversational AI interviewer using GPT-4.
    Manages the full interview session state and generates
    contextual, intelligent responses.
    """

    def __init__(self, api_key: str, role: str, hr_count: int, tech_count: int):
        self.client = OpenAI(api_key=api_key)
        self.state = InterviewerState(
            role=role,
            hr_count=hr_count,
            tech_count=tech_count
        )
        self.system_prompt = _build_system_prompt(role, hr_count, tech_count)
        self.total_questions = hr_count + tech_count

    def start_interview(self) -> str:
        """Generate the opening message from the interviewer."""
        opening = (
            f"Hello, and welcome. I'm glad you could join us today. "
            f"I'll be conducting your interview for the {ENGINEERING_ROLES.get(self.state.role, 'engineering')} role. "
            f"We'll cover both behavioural and technical areas. "
            f"Please answer as naturally as you would in a real interview — take your time. "
            f"Let's get started. {OPENING_QUESTIONS.get('hr', 'Tell me about yourself.')}"
        )
        self.state.turns.append(Turn(role="interviewer", text=opening, is_question=True))
        self.state.questions_asked += 1
        self.state.hr_asked += 1
        return opening

    def respond_to_answer(self, candidate_answer: str) -> str:
        """
        Process the candidate's answer and generate the interviewer's next response.
        Uses GPT-4 to decide whether to follow up or move to the next question.
        """
        # Add candidate turn
        self.state.turns.append(Turn(role="candidate", text=candidate_answer))

        # Check if we've covered enough questions
        total_asked = self.state.questions_asked
        is_last = total_asked >= self.total_questions

        # Build instruction for GPT based on state
        if is_last:
            instruction = (
                f"\n\n[INSTRUCTION — do not include this in your response]: "
                f"You have asked {total_asked} questions which is the target. "
                f"Give a brief warm closing remark acknowledging their last answer, "
                f"then end the interview with exactly: "
                f"'That brings us to the end of our interview. Thank you for your time.'"
            )
        elif self.state.consecutive_followups >= 2:
            # Force move to next topic
            next_q = _get_next_seed_question(self.state)
            instruction = (
                f"\n\n[INSTRUCTION — do not include this in your response]: "
                f"You have followed up enough on this topic. "
                f"Give a one-sentence acknowledgement of their answer, then ask this next question: "
                f"'{next_q}' — phrase it naturally in your own words."
            )
            self.state.consecutive_followups = 0
            self.state.questions_asked += 1
            if "tell me about" in next_q.lower() or "describe" in next_q.lower() or "time you" in next_q.lower():
                self.state.hr_asked += 1
            else:
                self.state.tech_asked += 1
        else:
            next_q = _get_next_seed_question(self.state)
            instruction = (
                f"\n\n[INSTRUCTION — do not include this in your response]: "
                f"Questions asked so far: {total_asked}/{self.total_questions}. "
                f"HR asked: {self.state.hr_asked}/{self.state.hr_count}. "
                f"Technical asked: {self.state.tech_asked}/{self.state.tech_count}. "
                f"Decide: if the answer was vague or very brief (under 3 sentences), ask a natural follow-up probing for more detail. "
                f"If the answer was complete, acknowledge briefly and ask this next question in your own natural words: '{next_q}'. "
                f"If you ask a follow-up, do NOT ask the next seed question yet — save it for after."
            )

        # Build messages with instruction appended to last candidate turn
        messages = _build_messages(self.state, self.system_prompt)
        messages[-1]["content"] += instruction

        # Call GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=300,
        )

        interviewer_text = response.choices[0].message.content.strip()

        # Detect if this was a follow-up or a new question
        is_followup = (
            not is_last and
            self.state.consecutive_followups < 2 and
            not any(kw in interviewer_text.lower() for kw in [
                "moving on", "let's move", "next question", "another area",
                "different topic", "let me ask you about"
            ])
        )

        if is_followup and not is_last:
            self.state.consecutive_followups += 1
        else:
            self.state.consecutive_followups = 0
            if not is_last:
                self.state.questions_asked += 1

        # Detect interview end
        if "that brings us to the end" in interviewer_text.lower():
            self.state.is_complete = True

        self.state.turns.append(Turn(
            role="interviewer",
            text=interviewer_text,
            is_question=True,
            is_followup=is_followup
        ))

        return interviewer_text

    def get_conversation_history(self) -> List[Turn]:
        return self.state.turns

    @property
    def is_complete(self) -> bool:
        return self.state.is_complete
