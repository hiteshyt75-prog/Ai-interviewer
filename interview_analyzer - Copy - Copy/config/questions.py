# config/questions.py
"""
Seed question bank per role. The AI will use these as starting points
and generate follow-ups dynamically based on the candidate's answers.
"""

OPENING_QUESTIONS = {
    "hr": "Let's start with a classic — tell me about yourself, your background, and what brings you here today.",
    "technical": "Let's dive in. Can you walk me through a recent technical project you worked on and the key decisions you made?"
}

SEED_QUESTIONS = {
    "software_engineer": [
        "Explain the difference between a stack and a queue and when you'd use each.",
        "How would you design a URL shortening service?",
        "Walk me through how you'd debug a performance issue in production.",
        "What's your approach to writing testable code?",
        "Explain a design pattern you've used and why you chose it.",
        "How do you handle technical debt in a fast-moving team?",
        "Describe the most complex system you've built or contributed to.",
        "How would you approach designing a system for one million concurrent users?",
    ],
    "data_scientist": [
        "Explain the bias-variance tradeoff and how it affects your modelling decisions.",
        "How do you handle class imbalance in a classification problem?",
        "Walk me through how you'd build a recommendation system from scratch.",
        "How do you decide which features to include in a model?",
        "Explain cross-validation and when you'd use different strategies.",
        "Describe a time your model performed well in testing but poorly in production.",
        "How do you communicate model results to non-technical stakeholders?",
        "What's your approach to A/B testing and statistical significance?",
    ],
    "mechanical_engineer": [
        "Walk me through your design process when starting a new component from scratch.",
        "Explain how you account for fatigue failure in mechanical design.",
        "Describe a time your design didn't perform as expected and how you resolved it.",
        "How do you approach finite element analysis in your work?",
        "Explain the tradeoffs between different material choices for a structural component.",
        "How do you ensure manufacturability in your designs?",
        "Describe your experience with tolerance stack-up analysis.",
        "How do you approach thermal management in mechanical systems?",
    ],
    "electrical_engineer": [
        "Walk me through how you would design a power supply circuit for an embedded system.",
        "How do you approach EMI/EMC compliance in PCB design?",
        "Describe a time you debugged a difficult circuit problem.",
        "Explain your approach to signal integrity in high-speed designs.",
        "How do you select components when designing for reliability?",
        "Describe your experience with microcontroller or FPGA programming.",
        "How do you validate an electrical design before production?",
        "Explain the tradeoffs between analog and digital control systems.",
    ],
    "civil_engineer": [
        "Walk me through how you approach a structural analysis for a new building.",
        "How do you account for seismic or wind loads in your designs?",
        "Describe the most challenging geotechnical problem you've worked on.",
        "How do you manage quality control on a large construction site?",
        "Explain your approach to stormwater or drainage design.",
        "How do you handle conflicts between design intent and site conditions?",
        "Describe your experience with BIM software and how it changed your workflow.",
        "How do you ensure safety compliance throughout a construction project?",
    ],
    "chemical_engineer": [
        "Walk me through how you would troubleshoot an underperforming reactor.",
        "How do you approach process scale-up from lab to plant?",
        "Describe your experience with process safety and hazard analysis.",
        "Explain how you would optimise a distillation column.",
        "How do you approach mass and energy balance for a new process?",
        "Describe a time a process didn't behave as expected and how you investigated it.",
        "How do you ensure regulatory compliance in process design?",
        "Explain the tradeoffs between batch and continuous processing.",
    ],
    "devops_engineer": [
        "Walk me through a CI/CD pipeline you've built from scratch.",
        "How do you approach infrastructure as code in a large organisation?",
        "Describe how you'd set up monitoring and alerting for a new microservice.",
        "How do you handle secrets management across multiple environments?",
        "Explain your approach to incident response and root cause analysis.",
        "Describe a time a deployment caused an outage and how you handled it.",
        "How do you balance velocity and reliability in a deployment pipeline?",
        "Explain your approach to container orchestration with Kubernetes.",
    ],
    "product_manager": [
        "Walk me through how you'd prioritise a backlog with competing stakeholder demands.",
        "Describe how you've used data to make a significant product decision.",
        "How do you define and measure success for a new product feature?",
        "Tell me about a product launch that didn't go as planned.",
        "How do you align engineering, design, and business around a product roadmap?",
        "Describe your approach to user research and how it's shaped a product decision.",
        "How do you handle a situation where the data and user feedback contradict each other?",
        "Explain how you'd approach entering a new market with an existing product.",
    ],
}

HR_SEED_QUESTIONS = [
    "Tell me about a time you had to resolve a significant conflict at work.",
    "Describe a situation where you failed and what you learned from it.",
    "Tell me about your greatest professional achievement.",
    "How do you handle working under significant pressure or tight deadlines?",
    "Describe a time you had to adapt quickly to a major unexpected change.",
    "Tell me about a time you had to lead without formal authority.",
    "How do you approach receiving difficult feedback?",
    "Describe a time you had to influence someone who disagreed with you.",
]

ENGINEERING_ROLES = {
    "software_engineer": "Software Engineer",
    "data_scientist": "Data Scientist",
    "mechanical_engineer": "Mechanical Engineer",
    "electrical_engineer": "Electrical Engineer",
    "civil_engineer": "Civil Engineer",
    "chemical_engineer": "Chemical Engineer",
    "devops_engineer": "DevOps Engineer",
    "product_manager": "Product Manager",
}
