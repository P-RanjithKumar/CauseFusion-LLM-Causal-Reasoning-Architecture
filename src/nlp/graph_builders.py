from typing import List, Dict, Tuple, Optional
import spacy
import spacy.cli
from core.causal_model import CausalModel

class Level1GraphBuilder:
    """Keyword-based domain classifier for routing to appropriate SCM."""

    DOMAIN_KEYWORDS = {
        "physical": [
            "gravity", "force", "friction", "break", "movement", "fall", "balance",
            "impact", "sound", "pressure", "mechanics", "spatial", "collision",
            "drop", "trip", "slip", "strike", "breaking", "falling"
        ],
        "biological": [
            "health", "virus", "bacteria", "germs", "pathogen", "disease", "illness",
            "sickness", "fever", "infection", "exercise", "workout", "nutrition",
            "diet", "digestion", "stomach", "injury", "wound", "bleed", "bleeding",
            "treatment", "medicine", "healing", "cure", "sleep", "recovery",
            "allergy", "sneeze", "sneezing"
        ],
        "environmental": [
            "weather", "storm", "disaster", "ecosystem", "rain", "wetness", "sun",
            "heat", "light", "shadow", "wind", "erosion", "drought", "fire",
            "earthquake", "landslide", "mudslide", "flood", "climate", "sunlight"
        ],
        "social": [
            "argument", "conflict", "friendship", "relationship", "communicate",
            "communication", "chat", "conversation", "befriend", "love", "hate",
            "trust", "distrust", "cooperation", "collaborate", "teamwork",
            "alliance", "law", "legal", "rules", "compliance", "obey", "insult",
            "pride", "embarrassment", "apology", "forgiveness", "arguing", "befriending"
        ],
        "cognitive": [
            "intention", "emotion", "mental", "confusion", "learn", "learning",
            "study", "fear", "threat", "anxiety", "nervous", "success", "failure",
            "confidence", "sadness", "joy", "excitement", "memory", "curiosity",
            "exploration", "concentration", "focus", "comprehension", "emotions",
            "intentions"
        ],
        "institutional": [
            "work", "economy", "economics", "structure", "organization", "hiring",
            "hire", "employ", "employment", "job", "career", "inflation", "banking",
            "bank", "price", "credit", "budget", "investment", "invest", "profit",
            "loss", "debt", "bankruptcy", "tax", "promotion", "merit", "deregulation",
            "competition", "scarcity"
        ]
    }

    def classify(self, premise: str, choice1: str = "", choice2: str = "") -> str:
        """Classify domain using ALL available text (premise + both choices).

        Enhanced V2: Uses word-boundary matching to prevent partial hits
        (e.g., 'daylight' no longer matches keyword 'light') and considers
        choice texts which often contain the most domain-specific language.
        """
        import re
        # Combine all available text for richer keyword matching
        text = f"{premise} {choice1} {choice2}".lower()
        words = set(re.findall(r'\b\w+\b', text))

        scores = {domain: 0 for domain in self.DOMAIN_KEYWORDS}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            for kw in keywords:
                if kw in words:  # Word-boundary match, not substring
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
        domain = self.l1.classify(premise, choice1, choice2)
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
