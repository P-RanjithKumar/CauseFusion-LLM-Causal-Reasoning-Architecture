from typing import List, Dict, Tuple, Optional
import spacy
import spacy.cli
from core.causal_model import CausalModel

class Level1GraphBuilder:
    """Keyword-based domain classifier for routing to appropriate SCM."""

    DOMAIN_KEYWORDS = {
        "physical": [
            "gravity", "heat", "cold", "force", "rain", "sun", "wind", "fire",
            "electricity", "pressure", "friction", "impact", "break", "melt",
            "freeze", "burn", "water", "light", "shadow", "movement"
        ],
        "social": [
            "stress", "anger", "sadness", "kindness", "gratitude", "fear",
            "success", "failure", "trust", "betrayal", "love", "hate",
            "jealousy", "pride", "embarrassment", "friendship", "conflict",
            "insult", "praise", "empathy", "guilt"
        ],
        "temporal": [
            "practice", "neglect", "repetition", "time", "growth", "decay",
            "habit", "improvement", "skill", "learning", "experience",
            "age", "maturity", "memory", "forgetting", "progress"
        ],
        "economic": [
            "income", "wealth", "debt", "expense", "profit", "loss", "payment",
            "discount", "price", "purchase", "poverty", "salary", "cost",
            "spend", "spending", "budget", "money", "financial", "investment",
            "tax", "credit", "loan", "employment", "unemployed", "job"
        ],
        "cognitive": [
            "study", "knowledge", "confusion", "investigation", "understanding",
            "mistake", "skill", "concentration", "ignorance", "misunderstanding",
            "question", "clarity", "explanation", "evidence", "belief", "reasoning",
            "conclusion", "observation", "hypothesis", "testing", "feedback",
            "improvement", "example", "comprehension", "pattern", "memory"
        ],
        "work": [
            "success", "promotion", "failure", "termination", "skill", "hiring",
            "mistake", "discipline", "innovation", "advancement", "performance",
            "evaluation", "teamwork", "project", "conflict", "competence",
            "negligence", "dedication", "reward", "experience", "training",
            "leadership", "communication", "effort", "accomplishment", "deadline",
            "pressure", "collaboration", "merit", "employment", "job", "career"
        ]
    }

    def classify(self, premise: str) -> str:
        text = premise.lower()
        scores = {domain: 0 for domain in self.DOMAIN_KEYWORDS}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    scores[domain] += 1
        
        # Return domain with highest score, default to physical
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "physical"


class Level2GraphBuilder:
    """spaCy NLP-based agent to extract causal variables from text."""

    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_md")
        except OSError:
            spacy.cli.download("en_core_web_md")
            self.nlp = spacy.load("en_core_web_md")

    def extract_variables(self, text: str) -> List[str]:
        doc = self.nlp(text)
        # Extract nouns and verbs as potential causal variables
        vars_found = [
            token.lemma_.lower() 
            for token in doc 
            if token.pos_ in ("NOUN", "VERB", "ADJ")
        ]
        return vars_found


class Orchestrator:
    """Main routing engine: Classifies domain → selects SCM."""

    def __init__(self, scms: Dict[str, CausalModel]):
        self.scms = scms
        self.l1 = Level1GraphBuilder()
        self.l2 = Level2GraphBuilder()

    def route(self, premise: str, choice1: str, choice2: str) -> Dict:
        domain = self.l1.classify(premise)
        scm = self.scms.get(domain, self.scms["physical"])
        
        # Try Level 2 extraction for variable matching
        p_vars = self.l2.extract_variables(premise)
        c1_vars = self.l2.extract_variables(choice1)
        c2_vars = self.l2.extract_variables(choice2)

        return {
            "domain": domain,
            "scm": scm,
            "premise_vars": p_vars,
            "choice1_vars": c1_vars,
            "choice2_vars": c2_vars,
            "method": "Level 1 (Keywords) + Level 2 (NLP)"
        }
