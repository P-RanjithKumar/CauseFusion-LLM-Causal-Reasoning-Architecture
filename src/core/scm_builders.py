from .causal_model import CausalModel

def build_physical_causation_model() -> CausalModel:
    m = CausalModel(domain="physical")
    edges = [
        ("gravity", "falling", 0.95),
        ("balance_loss", "falling", 0.90),
        ("force", "breaking", 0.85),
        ("force", "movement", 0.80),
        ("friction", "heat", 0.70),
        ("impact", "breaking", 0.90),
        ("impact", "sound", 0.80),
        ("pressure", "breaking", 0.75),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    
    m.add_synonyms("falling", ["fell", "fall", "trip", "tripped", "stumble", "stumbled", "tumble", "dropped", "slumped", "collapsed"])
    m.add_synonyms("balance_loss", ["balance", "unsteady", "dizzy", "stumble", "wobble"])
    m.add_synonyms("force", ["push", "pushed", "pull", "pulled", "hit", "struck", "threw", "throw", "kick", "kicked"])
    m.add_synonyms("breaking", ["broke", "break", "broken", "shatter", "shattered", "crack", "cracked", "smash", "smashed", "snap", "snapped"])
    m.add_synonyms("movement", ["move", "moved", "moving", "motion", "sway", "swayed", "drift", "drifted"])
    m.add_synonyms("friction", ["rubbing", "rub", "slide", "chafing"])
    m.add_synonyms("heat", ["hot", "warm", "heated", "temperature"])
    m.add_synonyms("sound", ["noise", "loud", "bang", "crash", "boom", "clatter"])
    m.add_synonyms("impact", ["collision", "collide", "collided", "crash", "crashed", "bang", "banged", "bump", "bumped"])
    m.add_synonyms("gravity", ["heavy", "weight", "weigh"])
    m.add_synonyms("pressure", ["compress", "compressed", "squeeze", "squeezed", "press", "pressed", "crush", "crushed"])
    return m

def build_biological_causation_model() -> CausalModel:
    m = CausalModel(domain="biological")
    edges = [
        ("pathogen", "disease", 0.90),
        ("infection", "fever", 0.85),
        ("exercise", "health", 0.80),
        ("nutrition", "digestion", 0.75),
        ("injury", "bleeding", 0.90),
        ("treatment", "healing", 0.85),
        ("poison", "sickness", 0.95),
        ("sleep", "recovery", 0.80),
        ("allergy", "sneezing", 0.85),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
        
    m.add_synonyms("pathogen", ["virus", "bacteria", "germs", "microbe", "bug"])
    m.add_synonyms("disease", ["illness", "sickness", "disorder", "ailment", "condition"])
    m.add_synonyms("infection", ["infected", "septic", "inflammation"])
    m.add_synonyms("fever", ["temperature", "hot", "chills", "feverish"])
    m.add_synonyms("exercise", ["workout", "running", "jogging", "gym", "training", "exertion"])
    m.add_synonyms("health", ["fitness", "wellness", "strength", "well-being"])
    m.add_synonyms("nutrition", ["diet", "food", "nutrients", "eating", "meal"])
    m.add_synonyms("digestion", ["digest", "metabolism", "stomach"])
    m.add_synonyms("injury", ["wound", "cut", "bruise", "hurt", "fracture", "damage"])
    m.add_synonyms("bleeding", ["bleed", "blood", "hemorrhage"])
    m.add_synonyms("treatment", ["medicine", "therapy", "surgery", "cure", "drug", "care"])
    m.add_synonyms("healing", ["heal", "recover", "mend", "improved", "cure"])
    m.add_synonyms("poison", ["toxin", "venom", "toxic", "contaminated"])
    m.add_synonyms("sickness", ["nausea", "vomiting", "ill", "sick"])
    m.add_synonyms("sleep", ["nap", "slumber", "rest", "sleeping"])
    m.add_synonyms("recovery", ["healing", "recovery", "recuperation"])
    m.add_synonyms("allergy", ["allergic", "reaction", "hypersensitivity"])
    m.add_synonyms("sneezing", ["sneeze", "cough", "runny nose"])
    return m

def build_environmental_causation_model() -> CausalModel:
    m = CausalModel(domain="environmental")
    edges = [
        ("sun", "light", 0.95),
        ("sun", "heat", 0.80),
        ("rain", "wetness", 0.95),
        ("wind", "erosion", 0.60),
        ("storm", "damage", 0.85),
        ("drought", "fire", 0.75),
        ("earthquake", "landslide", 0.80),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
        
    m.add_synonyms("sun", ["sunny", "sunshine", "sunlight", "sunrise", "sunset", "solar"])
    m.add_synonyms("light", ["bright", "lit", "glow", "glowing", "illuminate", "lamp", "shine", "shone", "shining"])
    m.add_synonyms("heat", ["hot", "warm", "heated", "temperature", "boiling", "sweltering", "scorching"])
    m.add_synonyms("rain", ["rained", "raining", "storm", "shower", "downpour", "drizzle", "pour", "poured", "precipitation"])
    m.add_synonyms("wetness", ["wet", "soaked", "damp", "moist", "drenched", "soggy", "puddle", "water", "flooded"])
    m.add_synonyms("wind", ["windy", "blew", "blow", "breeze", "gust", "gale", "breezy"])
    m.add_synonyms("erosion", ["erode", "eroded", "weathering", "washing away"])
    m.add_synonyms("storm", ["hurricane", "tornado", "tempest", "thunderstorm", "blizzard"])
    m.add_synonyms("damage", ["destruction", "ruin", "devastation", "wreckage"])
    m.add_synonyms("drought", ["dryness", "dry", "arid", "dehydration"])
    m.add_synonyms("fire", ["flame", "flames", "blaze", "blazing", "inferno", "bonfire"])
    m.add_synonyms("earthquake", ["quake", "tremor", "seismic"])
    m.add_synonyms("landslide", ["mudslide", "avalanche", "rockfall"])
    return m

def build_social_causation_model() -> CausalModel:
    m = CausalModel(domain="social")
    edges = [
        ("argument", "conflict", 0.85),
        ("kindness", "friendship", 0.80),
        ("betrayal", "distrust", 0.85),
        ("communication", "understanding", 0.80),
        ("cooperation", "alliance", 0.75),
        ("law", "compliance", 0.70),
        ("insult", "resentment", 0.80),
        ("apology", "forgiveness", 0.85),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
        
    m.add_synonyms("argument", ["argue", "argued", "fight", "fought", "disagree", "disagreed", "clash", "clashed"])
    m.add_synonyms("conflict", ["fight", "fought", "quarrel", "quarreled", "dispute", "disputed", "struggle"])
    m.add_synonyms("kindness", ["kind", "generous", "helpful", "caring", "compassionate", "gentle", "nice"])
    m.add_synonyms("friendship", ["friend", "friends", "buddy", "companion", "pal", "befriend", "befriended"])
    m.add_synonyms("betrayal", ["betrayed", "deceive", "deceived", "lie", "lied", "cheat", "cheated", "backstab"])
    m.add_synonyms("distrust", ["suspicion", "wary", "mistrust", "doubt"])
    m.add_synonyms("communication", ["talk", "talking", "speak", "speaking", "discuss", "discussion", "chat", "conversation"])
    m.add_synonyms("understanding", ["understand", "understood", "comprehend", "grasp", "clarity"])
    m.add_synonyms("cooperation", ["cooperate", "cooperated", "collaborate", "collaborated", "teamwork", "help", "helped"])
    m.add_synonyms("alliance", ["partnership", "union", "coalition", "league"])
    m.add_synonyms("law", ["legal", "rule", "regulation", "statute", "legislation"])
    m.add_synonyms("compliance", ["obey", "obeyed", "follow", "followed", "adherence"])
    m.add_synonyms("insult", ["insulted", "offend", "offended", "mock", "mocked", "ridicule", "ridiculed", "bully", "bullied"])
    m.add_synonyms("resentment", ["anger", "bitterness", "grudge", "animosity"])
    m.add_synonyms("apology", ["apologize", "apologized", "sorry"])
    m.add_synonyms("forgiveness", ["forgive", "forgave", "forgiven", "pardon", "reconcile"])
    return m

def build_cognitive_causation_model() -> CausalModel:
    m = CausalModel(domain="cognitive")
    edges = [
        ("confusion", "learning", 0.70),
        ("threat", "fear", 0.90),
        ("success", "confidence", 0.85),
        ("failure", "sadness", 0.80),
        ("stress", "anxiety", 0.85),
        ("curiosity", "exploration", 0.80),
        ("concentration", "comprehension", 0.85),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
        
    m.add_synonyms("confusion", ["confused", "bewildered", "perplexed", "puzzled", "unclear"])
    m.add_synonyms("learning", ["learn", "learned", "learnt", "study", "studied", "education", "knowledge", "skills"])
    m.add_synonyms("threat", ["danger", "hazard", "risk", "warning"])
    m.add_synonyms("fear", ["afraid", "scared", "frightened", "terrified", "panic", "panicked", "dread"])
    m.add_synonyms("success", ["succeed", "succeeded", "won", "win", "achieve", "accomplish"])
    m.add_synonyms("confidence", ["confident", "self-assured", "certain", "sure"])
    m.add_synonyms("failure", ["fail", "failed", "lose", "lost", "defeat"])
    m.add_synonyms("sadness", ["sad", "unhappy", "depressed", "miserable", "sorrowful", "cry", "crying", "tears"])
    m.add_synonyms("stress", ["stressed", "overwhelmed", "pressure", "tense", "tension", "strain"])
    m.add_synonyms("anxiety", ["anxious", "worried", "nervous", "uneasy"])
    m.add_synonyms("curiosity", ["curious", "interest", "inquisitive", "wonder"])
    m.add_synonyms("exploration", ["explore", "exploring", "investigate", "discover", "search"])
    m.add_synonyms("concentration", ["concentrate", "focus", "focused", "attention", "attentive"])
    m.add_synonyms("comprehension", ["understand", "understood", "comprehend", "grasp", "insight"])
    return m

def build_institutional_causation_model() -> CausalModel:
    m = CausalModel(domain="institutional")
    edges = [
        ("hiring", "employment", 0.90),
        ("inflation", "price_increase", 0.85),
        ("investment", "profit", 0.80),
        ("debt", "bankruptcy", 0.75),
        ("performance", "promotion", 0.85),
        ("merit", "reward", 0.80),
        ("deregulation", "competition", 0.70),
        ("scarcity", "value", 0.80),
        ("banking", "credit", 0.75),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
        
    m.add_synonyms("hiring", ["hired", "hire", "employ", "employed", "recruit", "recruited"])
    m.add_synonyms("employment", ["job", "jobs", "work", "career", "occupation"])
    m.add_synonyms("inflation", ["price hike", "rising prices", "cost of living"])
    m.add_synonyms("price_increase", ["expensive", "cost more", "markup", "surge"])
    m.add_synonyms("investment", ["invest", "invested", "investing", "portfolio", "stock", "capital"])
    m.add_synonyms("profit", ["gain", "earnings", "lucrative", "returns"])
    m.add_synonyms("debt", ["owed", "owe", "loan", "borrow", "borrowed", "liabilities"])
    m.add_synonyms("bankruptcy", ["bankrupt", "insolvent", "ruin", "liquidation"])
    m.add_synonyms("performance", ["productivity", "results", "evaluation", "output"])
    m.add_synonyms("promotion", ["promoted", "advance", "advanced", "career climb"])
    m.add_synonyms("merit", ["achievement", "qualification", "excellence"])
    m.add_synonyms("reward", ["bonus", "raise", "incentive", "recognition"])
    m.add_synonyms("deregulation", ["deregulated", "liberalization", "free market"])
    m.add_synonyms("competition", ["competitors", "rivalry", "market fight"])
    m.add_synonyms("scarcity", ["shortage", "rare", "limited supply", "deficit"])
    m.add_synonyms("value", ["worth", "price", "valuation", "cost"])
    m.add_synonyms("banking", ["bank", "banks", "banker", "financial institution", "deposits"])
    m.add_synonyms("credit", ["borrowing power", "loan limit", "credit score"])
    return m
