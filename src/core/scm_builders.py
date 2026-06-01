from .causal_model import CausalModel

def build_physical_causation_model() -> CausalModel:
    m = CausalModel(domain="physical_causation")
    edges = [
        ("gravity", "falling", 0.95),
        ("balance_loss", "falling", 0.90),
        ("heat", "melting", 0.85),
        ("heat", "burning", 0.80),
        ("heat", "evaporation", 0.75),
        ("cold", "freezing", 0.90),
        ("cold", "condensation", 0.70),
        ("force", "breaking", 0.85),
        ("force", "movement", 0.80),
        ("rain", "wetness", 0.95),
        ("wetness", "slipping", 0.70),
        ("sun", "light", 0.95),
        ("sun", "heat", 0.80),
        ("light_blocked", "shadow", 0.90),
        ("wind", "movement", 0.75),
        ("wind", "erosion", 0.60),
        ("fire", "burning", 0.95),
        ("fire", "heat", 0.90),
        ("electricity", "shock", 0.85),
        ("electricity", "light", 0.80),
        ("pressure", "breaking", 0.75),
        ("friction", "heat", 0.70),
        ("impact", "breaking", 0.90),
        ("impact", "sound", 0.80),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    # Synonym mappings: natural text → SCM variable
    m.add_synonyms("falling", ["fell", "fall", "trip", "tripped", "stumble", "stumbled", "tumble", "dropped", "slumped", "collapsed"])
    m.add_synonyms("balance_loss", ["balance", "unsteady", "dizzy", "stumble", "wobble"])
    m.add_synonyms("heat", ["hot", "warm", "heated", "temperature", "boiling", "sweltering", "scorching"])
    m.add_synonyms("melting", ["melt", "melted", "thaw", "thawed", "dissolve", "dissolved"])
    m.add_synonyms("burning", ["burn", "burned", "burnt", "scorched", "charred", "ignite", "ignited"])
    m.add_synonyms("cold", ["cool", "chilly", "frigid", "icy", "frozen"])
    m.add_synonyms("freezing", ["freeze", "froze", "frozen", "ice", "iced", "solidify"])
    m.add_synonyms("force", ["push", "pushed", "pull", "pulled", "hit", "struck", "threw", "throw", "kick", "kicked"])
    m.add_synonyms("breaking", ["broke", "break", "broken", "shatter", "shattered", "crack", "cracked", "smash", "smashed", "snap", "snapped"])
    m.add_synonyms("rain", ["rained", "raining", "storm", "shower", "downpour", "drizzle", "pour", "poured"])
    m.add_synonyms("wetness", ["wet", "soaked", "damp", "moist", "drenched", "soggy", "puddle", "water", "flooded"])
    m.add_synonyms("slipping", ["slip", "slipped", "slide", "slid", "skid", "skidded"])
    m.add_synonyms("sun", ["sunny", "sunshine", "sunlight", "sunrise", "sunset", "solar"])
    m.add_synonyms("light", ["bright", "lit", "glow", "glowing", "illuminate", "lamp", "shine", "shone", "shining"])
    m.add_synonyms("shadow", ["dark", "shade", "dim", "darkness", "cast"])
    m.add_synonyms("wind", ["windy", "blew", "blow", "breeze", "gust", "gale", "breezy"])
    m.add_synonyms("movement", ["move", "moved", "moving", "motion", "sway", "swayed", "drift", "drifted"])
    m.add_synonyms("fire", ["flame", "flames", "blaze", "blazing", "inferno", "bonfire"])
    m.add_synonyms("electricity", ["electric", "power", "plug", "outlet", "current", "wire", "wired"])
    m.add_synonyms("shock", ["shocked", "electrocuted", "zap", "zapped", "jolt", "jolted"])
    m.add_synonyms("sound", ["noise", "loud", "bang", "crash", "boom", "clatter"])
    m.add_synonyms("impact", ["collision", "collide", "collided", "crash", "crashed", "bang", "banged", "bump", "bumped"])
    m.add_synonyms("evaporation", ["evaporate", "evaporated", "steam", "vapor", "dry", "dried"])
    m.add_synonyms("gravity", ["heavy", "weight", "weigh"])
    m.add_synonyms("pressure", ["compress", "compressed", "squeeze", "squeezed", "press", "pressed", "crush", "crushed"])
    return m


def build_social_causation_model() -> CausalModel:
    m = CausalModel(domain="social_causation")
    edges = [
        ("stress", "anger", 0.85),
        ("failure", "sadness", 0.90),
        ("kindness", "gratitude", 0.80),
        ("fear", "avoidance", 0.75),
        ("success", "confidence", 0.90),
        ("betrayal", "anger", 0.85),
        ("betrayal", "sadness", 0.80),
        ("trust", "kindness", 0.70),
        ("love", "kindness", 0.90),
        ("hate", "anger", 0.95),
        ("jealousy", "anger", 0.80),
        ("jealousy", "envy", 0.90),
        ("envy", "anger", 0.70),
        ("success", "pride", 0.85),
        ("failure", "embarrassment", 0.80),
        ("embarrassment", "avoidance", 0.75),
        ("sympathy", "kindness", 0.70),
        ("empathy", "sympathy", 0.80),
        ("anger", "conflict", 0.90),
        ("conflict", "stress", 0.80),
        ("pride", "confidence", 0.70),
        ("sadness", "empathy", 0.60),
        ("loneliness", "sadness", 0.85),
        ("friendship", "trust", 0.80),
        ("rejection", "sadness", 0.85),
        ("praise", "pride", 0.80),
        # Additional social edges for better coverage
        ("insult", "anger", 0.85),
        ("compliment", "pride", 0.75),
        ("guilt", "sadness", 0.70),
        ("gratitude", "kindness", 0.65),
        ("excitement", "confidence", 0.60),
        ("boredom", "sadness", 0.50),
        ("surprise", "excitement", 0.70),
        ("disappointment", "sadness", 0.85),
        ("encouragement", "confidence", 0.80),
        ("criticism", "sadness", 0.75),
        ("apology", "gratitude", 0.70),
        ("argument", "anger", 0.85),
        ("cooperation", "trust", 0.75),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    # Synonym mappings
    m.add_synonyms("anger", ["angry", "mad", "furious", "enraged", "irritated", "annoyed", "outraged", "irate", "temper", "rage", "upset"])
    m.add_synonyms("sadness", ["sad", "unhappy", "depressed", "miserable", "sorrowful", "heartbroken", "grief", "grieving", "mourning", "cry", "cried", "crying", "wept", "tears"])
    m.add_synonyms("kindness", ["kind", "generous", "helpful", "caring", "compassionate", "gentle", "thoughtful", "considerate", "nice"])
    m.add_synonyms("gratitude", ["grateful", "thankful", "appreciative", "thanked", "appreciate", "appreciated"])
    m.add_synonyms("fear", ["afraid", "scared", "frightened", "terrified", "nervous", "anxious", "worried", "panic", "panicked", "dread", "dreaded"])
    m.add_synonyms("avoidance", ["avoid", "avoided", "evade", "evaded", "flee", "fled", "ran", "escape", "escaped", "hide", "hid", "hiding"])
    m.add_synonyms("success", ["succeed", "succeeded", "won", "win", "winning", "achieve", "achieved", "accomplish", "passed", "triumph"])
    m.add_synonyms("confidence", ["confident", "self-assured", "bold", "brave", "courageous", "assured", "certain"])
    m.add_synonyms("failure", ["fail", "failed", "lose", "lost", "losing", "defeat", "defeated", "flunk", "flunked"])
    m.add_synonyms("pride", ["proud", "boast", "boasted", "brag", "bragged", "triumph", "triumphant", "accomplished"])
    m.add_synonyms("embarrassment", ["embarrassed", "ashamed", "humiliated", "awkward", "shame", "shamed", "blush", "blushed"])
    m.add_synonyms("trust", ["trusted", "trusting", "rely", "relied", "depend", "depended", "faith", "faithful", "loyal"])
    m.add_synonyms("betrayal", ["betrayed", "deceive", "deceived", "lie", "lied", "cheat", "cheated", "backstab"])
    m.add_synonyms("love", ["loved", "loving", "adore", "adored", "cherish", "cherished", "affection"])
    m.add_synonyms("hate", ["hated", "despise", "despised", "loathe", "loathed", "detest", "detested"])
    m.add_synonyms("jealousy", ["jealous", "envious", "covet", "coveted", "possessive"])
    m.add_synonyms("empathy", ["empathetic", "understand", "understood", "relate", "compassion", "sympathize"])
    m.add_synonyms("conflict", ["fight", "fought", "quarrel", "quarreled", "dispute", "disputed", "argue", "argued"])
    m.add_synonyms("stress", ["stressed", "overwhelmed", "pressure", "pressured", "tense", "tension", "strain", "strained"])
    m.add_synonyms("loneliness", ["lonely", "alone", "isolated", "solitary", "abandoned"])
    m.add_synonyms("friendship", ["friend", "friends", "buddy", "companion", "pal", "ally"])
    m.add_synonyms("rejection", ["rejected", "refuse", "refused", "deny", "denied", "dismiss", "dismissed", "turned down"])
    m.add_synonyms("praise", ["praised", "compliment", "complimented", "commend", "commended", "applaud", "applauded"])
    m.add_synonyms("insult", ["insulted", "offend", "offended", "mock", "mocked", "ridicule", "ridiculed", "bully", "bullied"])
    m.add_synonyms("guilt", ["guilty", "remorse", "regret", "regretted", "sorry"])
    m.add_synonyms("disappointment", ["disappointed", "let down", "dissatisfied", "disheartened"])
    m.add_synonyms("encouragement", ["encouraged", "motivate", "motivated", "inspire", "inspired", "support", "supported"])
    m.add_synonyms("criticism", ["criticized", "critique", "blamed", "blame", "scolded", "scold", "reprimand"])
    m.add_synonyms("excitement", ["excited", "thrilled", "eager", "enthusiastic", "elated", "overjoyed", "happy", "delighted", "joy", "joyful", "pleased"])
    m.add_synonyms("surprise", ["surprised", "astonished", "amazed", "startled", "stunned", "unexpected"])
    m.add_synonyms("apology", ["apologize", "apologized", "sorry", "forgive", "forgave", "forgiven"])
    m.add_synonyms("cooperation", ["cooperate", "cooperated", "collaborate", "collaborated", "team", "helped"])
    m.add_synonyms("boredom", ["bored", "boring", "tedious", "dull", "uninterested", "monotonous"])
    m.add_synonyms("argument", ["argue", "argued", "fight", "fought", "disagree", "disagreed", "clash", "clashed"])
    m.add_synonyms("envy", ["envious", "jealous", "covet"])
    m.add_synonyms("sympathy", ["sympathetic", "pity", "pitied", "commiserate", "condolence"])
    m.add_synonyms("compliment", ["complimented", "flattery", "flattered", "admire", "admired"])
    return m


def build_temporal_causation_model() -> CausalModel:
    m = CausalModel(domain="temporal_causation")
    edges = [
        ("practice", "improvement", 0.90),
        ("neglect", "decay", 0.85),
        ("repetition", "habit", 0.80),
        ("time", "growth", 0.70),
        # Extended
        ("time", "experience", 0.80),
        ("experience", "learning", 0.85),
        ("learning", "skill", 0.90),
        ("skill", "improvement", 0.80),
        ("neglect", "forgetting", 0.75),
        ("time", "forgetting", 0.60),
        ("repetition", "memory", 0.70),
        ("memory", "learning", 0.80),
        ("time", "age", 1.00),
        ("age", "maturity", 0.70),
        ("practice", "skill", 0.85),
        ("improvement", "progress", 0.80),
        ("growth", "development", 0.90),
        ("waiting", "patience", 0.60),
        ("patience", "maturity", 0.65),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    # Synonym mappings
    m.add_synonyms("practice", ["practiced", "practised", "train", "trained", "training", "rehearse", "rehearsed", "exercise", "exercised", "study", "studied", "studying"])
    m.add_synonyms("improvement", ["improve", "improved", "improving", "better", "bettered", "enhanced", "progressed", "advancement"])
    m.add_synonyms("neglect", ["neglected", "neglecting", "ignore", "ignored", "abandon", "abandoned", "disregard", "disregarded"])
    m.add_synonyms("decay", ["decayed", "deteriorate", "deteriorated", "worsen", "worsened", "decline", "declined", "rot", "rotted"])
    m.add_synonyms("repetition", ["repeat", "repeated", "repeating", "routine", "regular", "daily", "consistently", "continuously"])
    m.add_synonyms("habit", ["habitual", "routine", "custom", "accustomed", "pattern", "ritual"])
    m.add_synonyms("time", ["years", "months", "weeks", "days", "hours", "long", "eventually", "gradually", "slowly", "soon", "later", "after", "period", "duration", "while", "season"])
    m.add_synonyms("growth", ["grow", "grew", "grown", "growing", "expand", "expanded", "increase", "increased"])
    m.add_synonyms("experience", ["experienced", "encountered", "exposure", "went through", "undergone"])
    m.add_synonyms("learning", ["learn", "learned", "learnt", "study", "studied", "education", "educated", "knowledge", "knew"])
    m.add_synonyms("skill", ["skilled", "skillful", "talented", "proficient", "expert", "expertise", "master", "mastered", "able", "ability", "capable"])
    m.add_synonyms("forgetting", ["forget", "forgot", "forgotten", "memory loss", "amnesia", "faded"])
    m.add_synonyms("memory", ["remember", "remembered", "recall", "recalled", "memorize", "memorized", "recollection"])
    m.add_synonyms("age", ["aged", "aging", "old", "older", "elderly", "senior", "young", "younger"])
    m.add_synonyms("maturity", ["mature", "matured", "wise", "wisdom", "adult", "grown-up", "responsible"])
    m.add_synonyms("progress", ["progressed", "advancing", "forward", "ahead", "development"])
    m.add_synonyms("development", ["develop", "developed", "developing", "evolve", "evolved", "evolution"])
    m.add_synonyms("waiting", ["wait", "waited", "patient", "patience", "endure", "endured"])
    m.add_synonyms("patience", ["patient", "waited", "persevere", "persevered", "tolerant"])
    return m


def build_economic_causation_model() -> CausalModel:
    """Economic/Financial causal domain.
    
    Models financial decisions, transactions, and economic consequences.
    Common in COPA: job pay, purchasing, financial struggles, investment.
    """
    m = CausalModel(domain="economic_causation")
    edges = [
        ("income", "wealth", 0.90),
        ("debt", "stress", 0.85),
        ("expense", "budget_strain", 0.80),
        ("profit", "investment", 0.85),
        ("loss", "caution", 0.80),
        ("payment", "relief", 0.75),
        ("discount", "purchase", 0.80),
        ("price_increase", "resistance", 0.75),
        ("poverty", "struggle", 0.90),
        ("wealth", "opportunity", 0.80),
        ("saving", "security", 0.85),
        ("spending", "pleasure", 0.70),
        ("unemployment", "hardship", 0.85),
        ("promotion", "income", 0.90),
        ("salary", "comfort", 0.75),
        ("cost", "hesitation", 0.70),
        ("value", "interest", 0.65),
        ("tax", "burden", 0.80),
        ("credit", "borrowing", 0.85),
        ("bankruptcy", "desperation", 0.90),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    # Synonym mappings
    m.add_synonyms("income", ["earn", "earned", "earning", "salary", "wage", "paid", "pay", "payment", "revenue", "earnings"])
    m.add_synonyms("wealth", ["rich", "wealthy", "prosperous", "money", "fortune", "affluent", "well-off"])
    m.add_synonyms("debt", ["owed", "owe", "loan", "borrow", "borrowed", "owing", "indebted", "credit card"])
    m.add_synonyms("stress", ["stressed", "stressed out", "anxious", "worried", "pressure", "strain", "tension"])
    m.add_synonyms("expense", ["cost", "spent", "spend", "spending", "price", "expensive", "charge", "charged", "bill"])
    m.add_synonyms("budget_strain", ["tight budget", "financial strain", "broke", "struggling", "shortage", "shortfall"])
    m.add_synonyms("profit", ["profit", "gain", "earnings", "proceeds", "revenue", "lucrative"])
    m.add_synonyms("investment", ["invest", "invested", "investing", "investor", "portfolio", "stock", "bond"])
    m.add_synonyms("loss", ["lost", "lose", "loss", "deficit", "negative", "down", "declined"])
    m.add_synonyms("caution", ["cautious", "careful", "wary", "hesitant", "conservative", "prudent", "cautiously"])
    m.add_synonyms("payment", ["paid", "pay", "payment", "settle", "settled", "installment", "due"])
    m.add_synonyms("relief", ["relieved", "relief", "satisfied", "happy", "content", "comfort"])
    m.add_synonyms("discount", ["discounted", "reduced", "reduction", "sale", "bargain", "cheaper", "discount", "offer"])
    m.add_synonyms("purchase", ["bought", "buy", "buying", "purchase", "purchased", "shop", "shopped", "acquire", "acquired"])
    m.add_synonyms("price_increase", ["expensive", "cost more", "price hike", "inflation", "markup", "surge"])
    m.add_synonyms("resistance", ["resist", "reluctant", "unwilling", "refuse", "opposed", "hesitant"])
    m.add_synonyms("poverty", ["poor", "impoverished", "destitute", "broke", "penniless", "poverty-stricken"])
    m.add_synonyms("struggle", ["struggle", "struggling", "fight", "fighting", "hardship", "difficult", "tough"])
    m.add_synonyms("opportunity", ["opportunity", "opportunity", "chance", "possibility", "potential", "access"])
    m.add_synonyms("saving", ["save", "saved", "savings", "accumulate", "accumulated", "reserve"])
    m.add_synonyms("security", ["secure", "secured", "safety", "protected", "cushion", "emergency fund"])
    m.add_synonyms("spending", ["spent", "spend", "spending", "expenditure", "purchase", "buy"])
    m.add_synonyms("pleasure", ["pleased", "pleased", "happy", "enjoyment", "satisfaction", "joy"])
    m.add_synonyms("unemployment", ["unemployed", "jobless", "laid off", "fired", "out of work", "dismissed"])
    m.add_synonyms("hardship", ["hardship", "difficulty", "struggle", "adversity", "trouble", "suffering"])
    m.add_synonyms("promotion", ["promoted", "promotion", "promoted", "advance", "advanced", "upgrade"])
    m.add_synonyms("salary", ["salary", "wage", "paycheck", "income", "earnings", "compensation"])
    m.add_synonyms("comfort", ["comfortable", "comfort", "ease", "relaxed", "content", "satisfied"])
    m.add_synonyms("cost", ["cost", "expense", "price", "expensive", "costly", "charge", "charged"])
    m.add_synonyms("hesitation", ["hesitant", "hesitate", "hesitated", "uncertain", "doubt", "pause"])
    m.add_synonyms("value", ["valuable", "value", "worth", "priced", "appreciated", "valued"])
    m.add_synonyms("interest", ["interested", "interest", "intrigued", "appeal", "appealing", "attractive"])
    m.add_synonyms("tax", ["tax", "taxed", "taxes", "fee", "levy", "deduction", "withhold"])
    m.add_synonyms("burden", ["burden", "burdened", "weight", "load", "heavy", "strain"])
    m.add_synonyms("credit", ["credit", "creditworthy", "credible", "loan", "borrow", "lending", "lender"])
    m.add_synonyms("borrowing", ["borrow", "borrowed", "loan", "lender", "creditor", "debt"])
    m.add_synonyms("bankruptcy", ["bankrupt", "bankruptcy", "insolvent", "financial ruin", "collapse"])
    m.add_synonyms("desperation", ["desperate", "desperate", "despair", "hopeless", "help", "aid"])
    return m


def build_cognitive_causation_model() -> CausalModel:
    """Cognitive/Learning causal domain.
    
    Models understanding, knowledge, problem-solving, and mental processes.
    Common in COPA: confusion, realization, learning, misunderstanding.
    """
    m = CausalModel(domain="cognitive_causation")
    edges = [
        ("study", "knowledge", 0.90),
        ("confusion", "investigation", 0.80),
        ("understanding", "confidence", 0.85),
        ("mistake", "learning", 0.80),
        ("practice", "skill", 0.85),
        ("concentration", "understanding", 0.80),
        ("ignorance", "misunderstanding", 0.85),
        ("question", "clarity", 0.75),
        ("explanation", "understanding", 0.85),
        ("evidence", "belief", 0.80),
        ("reasoning", "conclusion", 0.80),
        ("observation", "knowledge", 0.75),
        ("hypothesis", "testing", 0.75),
        ("feedback", "improvement", 0.80),
        ("example", "comprehension", 0.70),
        ("pattern", "recognition", 0.85),
        ("memory", "recall", 0.80),
        ("attention", "learning", 0.75),
        ("distraction", "confusion", 0.70),
        ("reflection", "insight", 0.75),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    # Synonym mappings
    m.add_synonyms("study", ["studied", "studying", "learn", "learned", "review", "reviewed", "read", "reading", "explore", "explored"])
    m.add_synonyms("knowledge", ["know", "knew", "known", "understand", "understood", "aware", "aware", "knowledgeable", "educated", "informed"])
    m.add_synonyms("confusion", ["confused", "confused", "confuse", "bewildered", "perplexed", "puzzled", "unclear", "muddy"])
    m.add_synonyms("investigation", ["investigate", "investigated", "inquire", "inquired", "research", "probe", "probed", "examine", "examined"])
    m.add_synonyms("understanding", ["understand", "understood", "comprehend", "comprehended", "grasp", "grasped", "get", "got", "insight", "clear"])
    m.add_synonyms("confidence", ["confident", "self-assured", "assured", "certain", "secure", "trusting", "belief", "sure", "convinced"])
    m.add_synonyms("mistake", ["mistake", "error", "wrong", "incorrect", "false", "mistaken", "wrong", "misunderstand", "misunderstood"])
    m.add_synonyms("learning", ["learn", "learned", "learnt", "education", "lesson", "study", "understand", "growth"])
    m.add_synonyms("practice", ["practice", "practiced", "practised", "train", "trained", "rehearse", "rehearsed", "exercise", "exercised"])
    m.add_synonyms("skill", ["skilled", "skillful", "capable", "able", "proficient", "competent", "expert", "talent", "talented"])
    m.add_synonyms("concentration", ["concentrate", "concentrated", "focus", "focused", "attentive", "attention", "mindful", "deliberate"])
    m.add_synonyms("ignorance", ["ignore", "ignored", "unaware", "unknowing", "oblivious", "clueless", "ignorant", "uneducated"])
    m.add_synonyms("misunderstanding", ["misunderstand", "misunderstood", "confused", "misconstrue", "misconstrued", "misinterpret", "misinterpreted"])
    m.add_synonyms("question", ["question", "questioned", "ask", "asked", "inquiry", "inquire", "puzzlement", "wonder", "wondered"])
    m.add_synonyms("clarity", ["clear", "clarity", "understood", "obvious", "evident", "transparent", "lucid", "plain", "obvious"])
    m.add_synonyms("explanation", ["explain", "explained", "clarify", "clarified", "describe", "described", "elaborate", "elaborated"])
    m.add_synonyms("evidence", ["evidence", "proof", "demonstrate", "demonstrated", "show", "showed", "proof", "testify", "testimony"])
    m.add_synonyms("belief", ["believe", "believed", "faith", "conviction", "convinced", "trusting", "trust", "principle"])
    m.add_synonyms("reasoning", ["reason", "reasoned", "logic", "logical", "rationale", "think", "thought", "argue", "argued"])
    m.add_synonyms("conclusion", ["conclude", "concluded", "conclusion", "end", "determine", "determined", "judgement", "verdict"])
    m.add_synonyms("observation", ["observe", "observed", "notice", "noticed", "see", "saw", "witness", "witnessed", "view", "perception"])
    m.add_synonyms("knowledge", ["know", "knew", "known", "fact", "information", "data", "expertise", "understanding"])
    m.add_synonyms("hypothesis", ["hypothesis", "theory", "idea", "assumption", "speculation", "conjecture", "postulate", "suppose", "supposed"])
    m.add_synonyms("testing", ["test", "tested", "try", "tried", "experiment", "experimented", "verify", "verified", "validate", "validated"])
    m.add_synonyms("feedback", ["feedback", "comment", "critique", "input", "response", "reaction", "suggestion"])
    m.add_synonyms("improvement", ["improve", "improved", "improving", "better", "progress", "advancement", "enhance", "enhanced"])
    m.add_synonyms("example", ["example", "instance", "illustration", "sample", "case", "specimen", "demonstration"])
    m.add_synonyms("comprehension", ["comprehend", "comprehended", "understand", "understood", "grasp", "grasped", "follow", "followed"])
    m.add_synonyms("pattern", ["pattern", "trend", "sequence", "cycle", "rhythm", "structure", "repetition"])
    m.add_synonyms("recognition", ["recognize", "recognized", "identify", "identified", "acknowledge", "acknowledged", "aware"])
    m.add_synonyms("memory", ["remember", "remembered", "recall", "recalled", "memorize", "memorized", "recollection", "mind"])
    m.add_synonyms("recall", ["recall", "remembered", "remember", "recollect", "recollected", "retrieve", "retrieved"])
    m.add_synonyms("attention", ["attention", "attend", "attended", "focus", "focused", "notice", "noticed", "heed", "heeded"])
    m.add_synonyms("distraction", ["distract", "distracted", "distraction", "interrupt", "interrupted", "divert", "diverted", "side-track"])
    m.add_synonyms("reflection", ["reflect", "reflected", "contemplate", "contemplated", "ponder", "pondered", "meditate", "meditated"])
    m.add_synonyms("insight", ["insight", "realization", "realized", "revelation", "epiphany", "understanding", "breakthrough", "aha"])
    return m


def build_work_causation_model() -> CausalModel:
    """Work/Professional causal domain.
    
    Models employment, career, performance, and professional relationships.
    Common in COPA: hiring, promotions, mistakes at work, team dynamics.
    """
    m = CausalModel(domain="work_causation")
    edges = [
        ("success", "promotion", 0.90),
        ("failure", "termination", 0.85),
        ("skill", "hiring", 0.85),
        ("mistake", "discipline", 0.80),
        ("innovation", "advancement", 0.80),
        ("performance", "evaluation", 0.85),
        ("teamwork", "project_success", 0.85),
        ("conflict", "tension", 0.80),
        ("competence", "respect", 0.80),
        ("negligence", "consequences", 0.85),
        ("dedication", "reward", 0.75),
        ("experience", "advancement", 0.80),
        ("training", "capability", 0.85),
        ("leadership", "followership", 0.75),
        ("communication", "clarity", 0.80),
        ("effort", "accomplishment", 0.75),
        ("deadline", "pressure", 0.85),
        ("collaboration", "synergy", 0.75),
        ("incompetence", "firing", 0.80),
        ("merit", "promotion", 0.85),
    ]
    for c, e, s in edges:
        m.add_causal_edge(c, e, s)
    # Synonym mappings
    m.add_synonyms("success", ["succeed", "succeeded", "successful", "accomplish", "accomplished", "achieve", "achieved", "win", "won"])
    m.add_synonyms("promotion", ["promoted", "promotion", "promoted", "advance", "advanced", "upgrade", "raise"])
    m.add_synonyms("failure", ["fail", "failed", "failure", "unsuccessful", "flop", "disappoint", "disappointed", "flopped"])
    m.add_synonyms("termination", ["fired", "fire", "fired", "terminated", "terminated", "laid off", "dismissed", "discharged"])
    m.add_synonyms("skill", ["skilled", "skillful", "capable", "competent", "proficient", "expert", "talent", "talented", "ability"])
    m.add_synonyms("hiring", ["hired", "hire", "hired", "employed", "employed", "recruit", "recruited", "selected"])
    m.add_synonyms("mistake", ["mistake", "error", "wrong", "incorrect", "slip", "miscalculation", "blunder", "oversight"])
    m.add_synonyms("discipline", ["disciplined", "discipline", "reprimand", "reprimanded", "corrected", "correct", "warning", "reprimanded"])
    m.add_synonyms("innovation", ["innovate", "innovative", "creative", "creativity", "novel", "new", "breakthrough", "invention"])
    m.add_synonyms("advancement", ["advanced", "advance", "advancement", "progress", "promotion", "move up", "climbing"])
    m.add_synonyms("performance", ["performance", "perform", "performed", "output", "productivity", "results", "delivery"])
    m.add_synonyms("evaluation", ["evaluate", "evaluated", "assessed", "assess", "review", "reviewed", "appraisal"])
    m.add_synonyms("teamwork", ["teamwork", "collaborate", "collaborated", "cooperate", "cooperated", "team", "together"])
    m.add_synonyms("project_success", ["project success", "successful project", "completion", "completed", "delivered", "success"])
    m.add_synonyms("conflict", ["conflict", "dispute", "disagreement", "tension", "clash", "fight", "fought", "argument", "argued"])
    m.add_synonyms("tension", ["tension", "strain", "stress", "stressed", "uncomfortable", "awkward", "friction"])
    m.add_synonyms("competence", ["competent", "competence", "capable", "ability", "skilled", "proficient", "qualified"])
    m.add_synonyms("respect", ["respect", "respected", "respectable", "admire", "admired", "esteem", "esteemed", "valued"])
    m.add_synonyms("negligence", ["negligent", "negligence", "careless", "careless", "neglect", "neglected", "irresponsible", "sloppy"])
    m.add_synonyms("consequences", ["consequence", "consequences", "result", "result", "outcome", "penalty", "repercussion"])
    m.add_synonyms("dedication", ["dedicated", "dedication", "committed", "commitment", "devoted", "devotion", "hardworking"])
    m.add_synonyms("reward", ["reward", "rewarded", "bonus", "incentive", "compensation", "benefit", "benefits", "recognition"])
    m.add_synonyms("experience", ["experience", "experienced", "veteran", "seasoned", "background", "exposure", "exposure"])
    m.add_synonyms("training", ["trained", "training", "trained", "education", "develop", "developed", "coaching"])
    m.add_synonyms("capability", ["capable", "capable", "ability", "aptitude", "proficiency", "competence", "skill"])
    m.add_synonyms("leadership", ["leadership", "lead", "led", "leader", "direct", "directed", "guide", "guided"])
    m.add_synonyms("followership", ["follow", "follower", "subordinate", "report", "reports to", "obey", "compliance"])
    m.add_synonyms("communication", ["communicate", "communicated", "message", "message", "inform", "informed", "discuss", "discussed"])
    m.add_synonyms("clarity", ["clear", "clarity", "understand", "understood", "obvious", "transparent", "lucid"])
    m.add_synonyms("effort", ["effort", "work", "worked", "hard work", "labor", "labor", "exertion", "endeavor"])
    m.add_synonyms("accomplishment", ["accomplish", "accomplished", "accomplishment", "completion", "completed", "success", "achievement"])
    m.add_synonyms("deadline", ["deadline", "due date", "deadline", "time limit", "schedule", "scheduled", "timeframe"])
    m.add_synonyms("pressure", ["pressure", "pressured", "stress", "stressed", "rush", "rushed", "urgent"])
    m.add_synonyms("collaboration", ["collaborate", "collaborated", "collaboration", "cooperate", "cooperated", "work together", "joint"])
    m.add_synonyms("synergy", ["synergy", "synergistic", "combined", "together", "cooperative", "effective", "powerful"])
    m.add_synonyms("incompetence", ["incompetent", "incompetence", "inability", "incapable", "unqualified", "inadequate"])
    m.add_synonyms("firing", ["fired", "fire", "fired", "terminated", "let go", "laid off", "discharge"])
    m.add_synonyms("merit", ["merit", "meritorious", "deserving", "worthy", "earned", "earned", "achievement", "accomplishment"])
    return m

