ASTA_DECISION_QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "What does the user want A.S.T.A. to do?",
        "criteria": {
            "conversation": "casual conversation, greeting, or social interaction",
            "command": "asking A.S.T.A. to perform an action on the computer",
            "memory": "asking A.S.T.A. to remember or recall information",
            "task": "asking A.S.T.A. to accomplish a multi-step goal",
            "question": "asking A.S.T.A. for information, explanation, analysis, or advice",
        },
    },
    "domain": {
        "type": "choice",
        "instructions": "What technical or interaction domain does the user request belong to?",
        "criteria": {
            "software": "programming, debugging, software engineering, architecture, repositories, code",
            "ai": "artificial intelligence, machine learning, LLMs, RAG, agents, model engineering",
            "electronics": "electronics, circuits, embedded systems, microcontrollers, hardware",
            "system": "operating system, applications, processes, files, screenshots, audio, computer control",
            "general": "general knowledge or a topic outside the other domains",
        },
    },
    "needs_tools": {
        "type": "noul",
        "instructions": "Does fulfilling the user request require A.S.T.A. to execute an external tool or computer action?",
    },
    "needs_reasoning": {
        "type": "noul",
        "instructions": "Does the request require multi-step technical reasoning rather than a simple factual response?",
    },
    "sensitivity": {
        "type": "score",
        "instructions": "How consequential or sensitive is the request for A.S.T.A. to handle?",
        "criteria": [
            "ordinary: low-consequence request",
            "moderate: some meaningful impact or private information",
            "high: significant financial, legal, safety, security, or system impact",
            "critical: potentially dangerous or highly privileged action",
        ],
    },
}
