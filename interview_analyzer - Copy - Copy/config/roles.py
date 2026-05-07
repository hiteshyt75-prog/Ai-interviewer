# config/roles.py
# Role-based configurations for technical interview analysis

ROLES = {
    "software_engineer": {
        "label": "Software Engineer",
        "keywords": [
            "algorithm", "data structure", "complexity", "big o", "recursion",
            "object oriented", "design pattern", "api", "rest", "database",
            "sql", "scalability", "testing", "git", "debugging", "abstraction",
            "inheritance", "polymorphism", "cache", "thread", "concurrency"
        ],
        "concepts": [
            "time complexity", "space complexity", "hash map", "binary tree",
            "linked list", "sorting", "searching", "dynamic programming",
            "microservices", "ci/cd", "unit test", "integration test"
        ],
        "expected_depth_keywords": ["because", "therefore", "which means", "for example", "trade-off"],
        "weight_overrides": {}
    },
    "data_scientist": {
        "label": "Data Scientist",
        "keywords": [
            "machine learning", "deep learning", "neural network", "model",
            "feature", "overfitting", "underfitting", "cross validation",
            "regression", "classification", "clustering", "statistics",
            "hypothesis", "p-value", "confidence interval", "bias", "variance",
            "gradient descent", "loss function", "accuracy", "precision", "recall"
        ],
        "concepts": [
            "train test split", "hyperparameter tuning", "feature engineering",
            "dimensionality reduction", "principal component analysis",
            "random forest", "gradient boosting", "natural language processing",
            "computer vision", "a/b testing", "data pipeline"
        ],
        "expected_depth_keywords": ["because", "therefore", "which means", "for example", "trade-off"],
        "weight_overrides": {}
    },
    "product_manager": {
        "label": "Product Manager",
        "keywords": [
            "user story", "roadmap", "stakeholder", "metric", "kpi", "okr",
            "priority", "backlog", "sprint", "agile", "scrum", "launch",
            "market", "customer", "feedback", "iteration", "mvp", "hypothesis",
            "data driven", "revenue", "retention", "engagement", "funnel"
        ],
        "concepts": [
            "minimum viable product", "user research", "go to market",
            "product market fit", "net promoter score", "monthly active users",
            "churn rate", "feature prioritization", "competitive analysis"
        ],
        "expected_depth_keywords": ["because", "therefore", "which means", "for example", "trade-off"],
        "weight_overrides": {}
    },
    "devops_engineer": {
        "label": "DevOps Engineer",
        "keywords": [
            "docker", "kubernetes", "container", "pipeline", "deployment",
            "infrastructure", "terraform", "ansible", "monitoring", "logging",
            "alerting", "load balancer", "auto scaling", "cloud", "aws", "gcp",
            "azure", "ci/cd", "jenkins", "github actions", "reliability", "sre"
        ],
        "concepts": [
            "continuous integration", "continuous deployment", "infrastructure as code",
            "service level objective", "mean time to recovery", "blue green deployment",
            "canary release", "observability", "incident management"
        ],
        "expected_depth_keywords": ["because", "therefore", "which means", "for example", "trade-off"],
        "weight_overrides": {}
    },
    "general": {
        "label": "General Technical",
        "keywords": [
            "system", "design", "architecture", "performance", "security",
            "scalability", "reliability", "maintainability", "efficiency",
            "optimization", "solution", "implementation", "approach"
        ],
        "concepts": [
            "best practice", "trade-off", "design decision", "technical debt",
            "code review", "documentation", "version control"
        ],
        "expected_depth_keywords": ["because", "therefore", "which means", "for example", "trade-off"],
        "weight_overrides": {}
    }
}

HR_COMPETENCIES = {
    "communication": {
        "keywords": ["explain", "communicate", "present", "discuss", "share", "inform", "listen"],
        "description": "Ability to clearly convey ideas"
    },
    "teamwork": {
        "keywords": ["team", "collaborate", "together", "colleague", "support", "help", "group", "partner"],
        "description": "Ability to work effectively in groups"
    },
    "leadership": {
        "keywords": ["lead", "mentor", "guide", "initiative", "responsibility", "decision", "manage", "influence"],
        "description": "Ability to lead and take ownership"
    },
    "problem_solving": {
        "keywords": ["solve", "analyze", "approach", "strategy", "solution", "identify", "resolve", "tackle"],
        "description": "Structured approach to challenges"
    },
    "adaptability": {
        "keywords": ["adapt", "flexible", "change", "learn", "adjust", "pivot", "overcome", "challenge"],
        "description": "Ability to handle change and ambiguity"
    },
    "conflict_resolution": {
        "keywords": ["conflict", "disagree", "tension", "resolve", "mediate", "compromise", "address", "differing"],
        "description": "Ability to manage and resolve conflicts"
    }
}

SCORING_WEIGHTS = {
    "hr": {
        "relevance": 0.25,
        "structure": 0.20,
        "competency_coverage": 0.25,
        "specificity": 0.20,
        "fluency": 0.10
    },
    "technical": {
        "relevance": 0.20,
        "keyword_coverage": 0.25,
        "depth": 0.25,
        "structure": 0.15,
        "fluency": 0.15
    }
}
