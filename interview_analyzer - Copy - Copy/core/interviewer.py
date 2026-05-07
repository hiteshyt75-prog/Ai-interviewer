# core/interviewer.py
"""
AI Interviewer powered by GPT-4o-mini.
- Never introduces itself with a name
- High temperature + shuffled topics = genuinely different questions every session
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional
from openai import OpenAI
from config.questions import ENGINEERING_ROLES


@dataclass
class Turn:
    role: str           # "interviewer" or "candidate"
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
    consecutive_followups: int = 0
    is_complete: bool = False


HR_TOPICS = [
    "a conflict with a colleague and how you resolved it",
    "a time you failed professionally and what changed after",
    "your most significant career achievement so far",
    "how you manage pressure and competing deadlines",
    "a time you had to adapt quickly to unexpected change",
    "a situation where you led without being the official manager",
    "how you respond when you receive criticism or hard feedback",
    "a time you changed someone's mind who strongly disagreed with you",
    "working through a period when your team was underperforming",
    "going well beyond what was expected of you on a project",
    "handling a disagreement with someone senior to you",
    "what genuinely motivates you day to day in your work",
    "a time you had to make a decision without all the information",
    "balancing quality versus speed when both matter",
    "dealing with a difficult stakeholder or client",
]

TECHNICAL_TOPICS = {
    "software_engineer": [
        "how you think about time and space complexity when writing code",
        "your approach to object oriented design in a real project",
        "designing a backend system that needs to scale to millions of users",
        "how you track down a bug in a live production system",
        "your philosophy on writing testable, maintainable code",
        "choosing between relational and non-relational databases for a given problem",
        "how REST APIs should be designed and what makes them RESTful",
        "handling concurrency and race conditions in your code",
        "a design pattern you have actually used and why you chose it",
        "your experience with CI/CD and automating deployments",
        "when and how you would introduce caching into a system",
        "how you manage technical debt in a fast moving codebase",
        "explaining how the internet works when a user types a URL",
        "how you approach code reviews — giving and receiving",
        "the tradeoffs between monolithic and microservice architectures",
    ],
    "data_scientist": [
        "how you think about the bias variance tradeoff in practice",
        "techniques you use to prevent overfitting in your models",
        "the difference between supervised and unsupervised approaches and when to use each",
        "your approach to cross validation and why it matters",
        "how ensemble methods like random forests or boosting actually work",
        "choosing the right evaluation metric for a classification problem",
        "how you handle missing data or severe class imbalance in a dataset",
        "explaining gradient descent to someone non-technical",
        "how you decide which features to keep or drop from a model",
        "designing an A/B test and interpreting its results",
        "your experience taking a model from experiment to production",
        "how neural networks learn and where they fail",
        "building a recommendation system from scratch",
        "communicating model results and uncertainty to a business audience",
        "the difference between correlation and causation in practice",
    ],
    "mechanical_engineer": [
        "walking through your design process from brief to final component",
        "how you account for fatigue and failure modes in a design",
        "your experience using finite element analysis and its limitations",
        "applying thermodynamic principles to a real engineering problem",
        "how you approach fluid flow calculations in a system",
        "designing for manufacturability and assembly",
        "how you handle tolerance stack-up in an assembly",
        "a time a design failed or underperformed and how you investigated it",
        "your experience with CAD tools and simulation software",
        "how PID control systems work and when you have applied them",
        "the tradeoffs between different material choices for a structural part",
        "ensuring safety factors are appropriate without overengineering",
        "your approach to writing engineering calculations and documentation",
        "how you validate a design before it goes to production",
        "root cause analysis after a mechanical failure",
    ],
    "electrical_engineer": [
        "designing a power supply circuit for an embedded application",
        "how you approach EMI and EMC compliance in a PCB layout",
        "your process for debugging a circuit that is not behaving as expected",
        "signal integrity challenges in high speed digital designs",
        "how you select components when designing for long term reliability",
        "your experience with microcontrollers or FPGAs in a real project",
        "validating an electrical design through simulation and testing",
        "tradeoffs between analog and digital control approaches",
        "how MOSFET transistors work and where you have used them",
        "three phase power systems and their practical applications",
        "how Fourier analysis applies to signal processing problems",
        "protecting a circuit against overcurrent or transient events",
        "your approach to writing firmware for embedded systems",
        "grounding and shielding strategies in sensitive electronics",
        "how you manage a PCB design through to manufacture",
    ],
    "civil_engineer": [
        "how you approach a structural analysis for a new building",
        "accounting for seismic or wind loads in your designs",
        "a challenging geotechnical problem you have worked through",
        "managing quality control across a large construction site",
        "your approach to stormwater drainage design",
        "handling a conflict between design intent and site conditions",
        "how BIM software has changed the way you work",
        "ensuring safety compliance throughout a project lifecycle",
        "the difference between ultimate limit state and serviceability checks",
        "how you design foundations for difficult ground conditions",
        "truss and beam analysis in structural design",
        "your experience with project scheduling and critical path methods",
        "working with contractors and managing construction interfaces",
        "how you approach a structural assessment of an existing building",
        "environmental considerations in civil infrastructure projects",
    ],
    "chemical_engineer": [
        "troubleshooting a reactor that is underperforming against design",
        "your approach to scaling a process from laboratory to plant scale",
        "how you approach process hazard analysis and safety cases",
        "optimising a distillation column for purity and energy efficiency",
        "setting up a mass and energy balance for a new process",
        "a time a process behaved unexpectedly and how you investigated it",
        "ensuring regulatory compliance in a process design project",
        "the tradeoffs between batch and continuous processing for a given product",
        "how catalysts work and how you select one for a reaction",
        "applying six sigma or lean principles to process improvement",
        "heat exchanger design and selection for a process application",
        "how you approach pump sizing and hydraulic calculations",
        "process safety management and permit to work systems",
        "your experience reading and creating P and ID diagrams",
        "how you handle hazardous materials safely in a plant environment",
    ],
    "devops_engineer": [
        "designing and building a CI/CD pipeline from scratch",
        "your approach to infrastructure as code across environments",
        "setting up monitoring alerting and observability for a new service",
        "how you manage secrets and credentials securely",
        "your process for incident response and root cause analysis",
        "a deployment that caused an outage and how you handled it",
        "balancing deployment velocity with system reliability",
        "Kubernetes orchestration in a production environment",
        "the difference between blue green and canary release strategies",
        "your approach to container security and vulnerability scanning",
        "how GitOps changes the deployment workflow",
        "auto scaling strategies and when they fail",
        "chaos engineering and why you would deliberately break things",
        "how you approach database migrations without downtime",
        "managing multi environment configuration and drift",
    ],
    "product_manager": [
        "how you prioritise a backlog when stakeholders all have conflicting priorities",
        "using data to drive a significant product decision",
        "defining and tracking the right success metrics for a feature",
        "a product launch that did not go as planned and what you did",
        "aligning engineering design and business around a roadmap",
        "conducting user research and translating findings into product decisions",
        "handling a situation where data and user feedback point in opposite directions",
        "entering a new market or launching a product in a new segment",
        "working through a disagreement between engineering and design",
        "how you define minimum viable product for a new initiative",
        "managing technical debt requests alongside feature work",
        "how you communicate product vision to different audiences",
        "running effective sprint planning and retrospective sessions",
        "using net promoter score or similar metrics to inform decisions",
        "how you decide a product or feature is ready to ship",
    ],
}

OPENER_STYLES = [
    "Walk me through",
    "Can you describe",
    "Give me an example of",
    "How would you approach",
    "Tell me about a time when",
    "What's your experience with",
    "Imagine you're faced with",
    "How do you typically handle",
    "What would you do if",
    "Describe a situation where",
    "In your experience, how do you",
    "How have you dealt with",
]


def _build_system_prompt(role: str, hr_count: int, tech_count: int) -> str:
    role_label = ENGINEERING_ROLES.get(role, role.replace("_", " ").title())
    total = hr_count + tech_count

    # Shuffle topics so GPT sees a different order every session
    tech_pool = TECHNICAL_TOPICS.get(role, list(TECHNICAL_TOPICS.values())[0]).copy()
    hr_pool = HR_TOPICS.copy()
    random.shuffle(tech_pool)
    random.shuffle(hr_pool)

    # Give GPT a random subset so it can't just repeat the same ones
    tech_sample = tech_pool[:min(8, len(tech_pool))]
    hr_sample = hr_pool[:min(6, len(hr_pool))]
    opener_sample = random.sample(OPENER_STYLES, 6)

    return f"""You are an AI interviewer conducting a live mock job interview for a {role_label} position.

IMPORTANT — WHO YOU ARE:
- You are an AI interviewer. Do NOT give yourself a human name. Do NOT say "I'm [Your Name]" or introduce yourself with any name.
- You may say "I'm your AI interviewer today" or simply start with the interview.
- Be professional, warm, and conversational — but clearly an AI, not a human pretending to have a name.

INTERVIEW STRUCTURE:
- Total: {total} questions ({hr_count} HR/behavioural + {tech_count} technical)
- Ask ONE question per turn. Never ask two questions in the same message.

TOPIC INSPIRATION (use these as loose guides — never copy them verbatim):
HR topics: {", ".join(hr_sample)}
Technical topics: {", ".join(tech_sample)}

QUESTION VARIETY — THIS IS CRITICAL:
- Every question MUST use a different opening phrase. Rotate through: {", ".join(opener_sample)} and others.
- NEVER start two questions with the same words.
- Rephrase topics completely in your own conversational words every time.
- Each question must feel genuinely different — vary length, angle, and specificity.
- Do not cluster all HR or all technical questions together — interleave them naturally.

FOLLOW-UP BEHAVIOUR:
- If an answer is vague or very brief, ask ONE short natural follow-up that references what they said.
- Maximum 2 follow-ups on the same topic, then move on.
- Follow-ups must directly reference something specific the candidate just said.

CLOSING:
- When you have covered enough questions, say EXACTLY:
  "That brings us to the end of our interview. Thank you for your time today."
  Nothing else.

FORMAT:
- Only output what you would say out loud. No labels, no markdown, no stage directions.
- Keep responses concise — you are an interviewer not a teacher.
"""


def _build_messages(state: InterviewerState, system_prompt: str) -> list:
    messages = [{"role": "system", "content": system_prompt}]
    for turn in state.turns:
        role = "assistant" if turn.role == "interviewer" else "user"
        messages.append({"role": role, "content": turn.text})
    return messages


class AIInterviewer:
    def __init__(self, api_key: str, role: str, hr_count: int, tech_count: int):
        self.client = OpenAI(api_key=api_key)
        self.state = InterviewerState(
            role=role,
            hr_count=hr_count,
            tech_count=tech_count
        )
        # Rebuild system prompt fresh each session — topics are reshuffled
        self.system_prompt = _build_system_prompt(role, hr_count, tech_count)
        self.total_questions = hr_count + tech_count

    def start_interview(self) -> str:
        """GPT generates a unique opening — never the same twice."""
        role_label = ENGINEERING_ROLES.get(self.state.role, "engineering")

        # Random seed instruction so even the opener varies
        opener_hint = random.choice([
            "ask about a recent project they found rewarding",
            "ask what drew them to this field or role",
            "ask about a technical challenge they recently solved",
            "ask about a career moment they are particularly proud of",
            "ask what kind of work energises them professionally",
        ])

        opener_instruction = (
            f"Start the interview now. "
            f"Greet the candidate briefly and introduce yourself as their AI interviewer for the {role_label} role — "
            f"do NOT give yourself a human name. "
            f"Mention the session covers both behavioural and technical areas. "
            f"Then immediately {opener_hint}. "
            f"Keep the whole opening under 4 sentences. Sound natural and human, not robotic."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": opener_instruction}
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=1.0,   # Max variety for openers
            max_tokens=180,
        )

        opening = response.choices[0].message.content.strip()
        self.state.turns.append(Turn(role="interviewer", text=opening, is_question=True))
        self.state.questions_asked += 1
        self.state.hr_asked += 1
        return opening

    def respond_to_answer(self, candidate_answer: str) -> str:
        """Process candidate answer and generate next interviewer turn."""
        self.state.turns.append(Turn(role="candidate", text=candidate_answer))

        total_asked = self.state.questions_asked
        is_last = total_asked >= self.total_questions
        hr_remaining = self.state.hr_count - self.state.hr_asked
        tech_remaining = self.state.tech_count - self.state.tech_asked

        if is_last:
            instruction = (
                "\n\n[SYSTEM — omit from response]: "
                f"You have reached {total_asked} questions. "
                "Acknowledge their last answer briefly, then close with exactly: "
                "'That brings us to the end of our interview. Thank you for your time today.'"
            )
        elif self.state.consecutive_followups >= 2:
            next_type = "HR/behavioural" if hr_remaining > tech_remaining else "technical"
            random_opener = random.choice(OPENER_STYLES)
            instruction = (
                "\n\n[SYSTEM — omit from response]: "
                f"You have followed up enough on this topic. Acknowledge briefly in one sentence, "
                f"then ask a completely fresh {next_type} question on a new topic. "
                f"Start with '{random_opener}' or a similar varied opener. "
                f"Do not reuse any phrasing from previous questions."
            )
            self.state.consecutive_followups = 0
            self.state.questions_asked += 1
            if next_type == "HR/behavioural":
                self.state.hr_asked += 1
            else:
                self.state.tech_asked += 1
        else:
            next_type = "HR/behavioural" if hr_remaining > tech_remaining else "technical"
            random_opener = random.choice(OPENER_STYLES)
            instruction = (
                "\n\n[SYSTEM — omit from response]: "
                f"Questions so far: {total_asked}/{self.total_questions}. "
                f"HR remaining: {hr_remaining}, Technical remaining: {tech_remaining}. "
                f"If the candidate's answer was vague or very short (under 3 sentences), "
                f"ask a short natural follow-up that references something specific they just said. "
                f"Otherwise acknowledge in one sentence and ask a fresh {next_type} question on a new topic — "
                f"start with '{random_opener}' or another varied opener you haven't used yet. "
                f"Never reuse phrasing from earlier questions."
            )

        messages = _build_messages(self.state, self.system_prompt)
        messages[-1]["content"] += instruction

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.95,
            max_tokens=250,
        )

        interviewer_text = response.choices[0].message.content.strip()

        # Detect whether this is a follow-up
        followup_signals = [
            "can you elaborate", "tell me more", "what do you mean",
            "how did that", "what happened next", "why did you",
            "could you expand", "what was the outcome", "how did you handle that specifically",
            "what specifically", "can you give me more detail", "and then what"
        ]
        is_followup = (
            not is_last
            and self.state.consecutive_followups < 2
            and any(sig in interviewer_text.lower() for sig in followup_signals)
        )

        if is_followup:
            self.state.consecutive_followups += 1
        else:
            self.state.consecutive_followups = 0
            if not is_last:
                self.state.questions_asked += 1
                if hr_remaining > tech_remaining:
                    self.state.hr_asked += 1
                else:
                    self.state.tech_asked += 1

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
