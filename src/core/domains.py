"""Five hand-written domain graphs.

Fixed (not learned) so a downstream accuracy change is attributable to the SCM
rather than to a structure-learning step. Each spec is a list of
(cause, effect, strength, polarity) edges plus a surface-word alias table.
"""
from __future__ import annotations

from typing import Dict, List

from .scm import CausalModel

# (cause, effect, strength, polarity), then alias lists per node.
_SPECS = {
    "physical": {
        "edges": [
            ("wet_surface", "slipped", 0.85, +1),
            ("slipped", "lost_balance", 0.80, +1),
            ("lost_balance", "fell", 0.90, +1),
            ("ice", "slipped", 0.80, +1),
            ("push", "fell", 0.70, +1),
            ("fell", "injury", 0.65, +1),
            ("impact", "broke", 0.85, +1),
            ("dropped", "broke", 0.80, +1),
            ("heavy_load", "broke", 0.60, +1),
            ("friction", "heat", 0.70, +1),
            ("heat", "melted", 0.80, +1),
            ("fire", "burned", 0.90, +1),
            ("support", "fell", 0.75, -1),
            ("grip", "slipped", 0.65, -1),
        ],
        "alias": {
            "wet_surface": ["wet", "water", "spill", "puddle", "damp", "slippery"],
            "slipped": ["slip", "slipped", "skid", "skidded"],
            "lost_balance": ["balance", "unsteady", "wobble", "stumble", "stumbled"],
            "fell": ["fall", "fell", "falling", "tumble", "tumbled", "collapsed"],
            "ice": ["ice", "icy", "frozen", "frost"],
            "push": ["push", "pushed", "shove", "shoved"],
            "injury": ["injury", "hurt", "bruise", "wound", "injured"],
            "impact": ["impact", "collision", "crash", "crashed", "hit", "struck"],
            "dropped": ["drop", "dropped", "fell off"],
            "broke": ["broke", "break", "broken", "shatter", "shattered", "snap"],
            "heavy_load": ["heavy", "weight", "load", "overloaded"],
            "friction": ["friction", "rubbing", "rub"],
            "heat": ["heat", "hot", "warm", "heated"],
            "melted": ["melt", "melted", "thaw"],
            "fire": ["fire", "flame", "flames", "blaze"],
            "burned": ["burn", "burned", "burnt", "scorched"],
            "support": ["support", "brace", "prop", "held"],
            "grip": ["grip", "traction", "tread"],
        },
    },
    "emotional": {
        "edges": [
            ("threat", "fear", 0.90, +1),
            ("fear", "avoidance", 0.70, +1),
            ("loss", "sadness", 0.85, +1),
            ("success", "joy", 0.85, +1),
            ("success", "confidence", 0.80, +1),
            ("failure", "sadness", 0.75, +1),
            ("failure", "confidence", 0.70, -1),
            ("stress", "anxiety", 0.85, +1),
            ("rest", "stress", 0.70, -1),
            ("praise", "confidence", 0.75, +1),
            ("surprise", "shock", 0.70, +1),
            ("anxiety", "avoidance", 0.60, +1),
        ],
        "alias": {
            "threat": ["threat", "danger", "risk", "menace"],
            "fear": ["fear", "afraid", "scared", "frightened", "terrified"],
            "avoidance": ["avoid", "avoided", "flee", "fled", "hide", "hid"],
            "loss": ["loss", "lost", "bereavement", "grief"],
            "sadness": ["sad", "sadness", "unhappy", "cried", "crying", "depressed"],
            "success": ["success", "succeeded", "won", "win", "achieved"],
            "joy": ["joy", "happy", "happiness", "delighted", "elated"],
            "confidence": ["confidence", "confident", "assured", "self-assured"],
            "failure": ["failure", "failed", "lost", "defeat", "defeated"],
            "stress": ["stress", "stressed", "pressure", "overwhelmed"],
            "anxiety": ["anxiety", "anxious", "worried", "nervous"],
            "rest": ["rest", "relax", "relaxed", "calm", "slept"],
            "praise": ["praise", "praised", "complimented", "applauded"],
            "surprise": ["surprise", "surprised", "unexpected"],
            "shock": ["shock", "shocked", "stunned"],
        },
    },
    "social": {
        "edges": [
            ("argument", "conflict", 0.85, +1),
            ("insult", "resentment", 0.80, +1),
            ("betrayal", "distrust", 0.85, +1),
            ("apology", "forgiveness", 0.80, +1),
            ("kindness", "friendship", 0.80, +1),
            ("kindness", "distrust", 0.70, -1),
            ("cooperation", "trust", 0.75, +1),
            ("lie", "distrust", 0.80, +1),
            ("gift", "gratitude", 0.75, +1),
            ("rudeness", "conflict", 0.70, +1),
            ("communication", "understanding", 0.75, +1),
            ("understanding", "conflict", 0.65, -1),
        ],
        "alias": {
            "argument": ["argue", "argued", "argument", "quarrel", "quarrelled", "dispute"],
            "conflict": ["conflict", "fight", "fought", "clash", "feud"],
            "insult": ["insult", "insulted", "mocked", "ridiculed", "offended"],
            "resentment": ["resentment", "bitterness", "grudge", "anger"],
            "betrayal": ["betray", "betrayed", "deceived", "cheated"],
            "distrust": ["distrust", "mistrust", "suspicion", "wary", "doubt"],
            "apology": ["apology", "apologise", "apologized", "sorry"],
            "forgiveness": ["forgive", "forgave", "forgiveness", "pardon"],
            "kindness": ["kind", "kindness", "caring", "generous", "helpful"],
            "friendship": ["friend", "friendship", "befriended", "companion"],
            "cooperation": ["cooperate", "cooperation", "collaborate", "teamwork"],
            "trust": ["trust", "trusted", "reliance"],
            "lie": ["lie", "lied", "dishonest", "falsehood"],
            "gift": ["gift", "present", "gave"],
            "gratitude": ["gratitude", "grateful", "thankful", "thanked"],
            "rudeness": ["rude", "rudeness", "impolite", "disrespect"],
            "communication": ["talk", "talked", "spoke", "discuss", "communication"],
            "understanding": ["understand", "understood", "understanding", "empathy"],
        },
    },
    "natural": {
        "edges": [
            ("rain", "wetness", 0.95, +1),
            ("rain", "flood", 0.65, +1),
            ("storm", "damage", 0.85, +1),
            ("sun", "heat", 0.80, +1),
            ("sun", "light", 0.95, +1),
            ("drought", "fire", 0.70, +1),
            ("wind", "erosion", 0.55, +1),
            ("earthquake", "landslide", 0.75, +1),
            ("frost", "ice", 0.85, +1),
            ("rain", "drought", 0.80, -1),
            ("cloud", "light", 0.60, -1),
        ],
        "alias": {
            "rain": ["rain", "rained", "raining", "downpour", "shower", "drizzle"],
            "wetness": ["wet", "wetness", "soaked", "damp", "drenched"],
            "flood": ["flood", "flooded", "flooding", "deluge"],
            "storm": ["storm", "hurricane", "tornado", "thunderstorm", "gale"],
            "damage": ["damage", "destruction", "wreckage", "ruined"],
            "sun": ["sun", "sunny", "sunshine", "sunlight", "solar"],
            "heat": ["heat", "hot", "warm", "sweltering"],
            "light": ["light", "bright", "daylight", "glow"],
            "drought": ["drought", "dry", "arid", "dryness"],
            "fire": ["fire", "wildfire", "blaze", "flames"],
            "wind": ["wind", "windy", "gust", "breeze"],
            "erosion": ["erosion", "eroded", "weathering"],
            "earthquake": ["earthquake", "quake", "tremor", "seismic"],
            "landslide": ["landslide", "mudslide", "rockfall", "avalanche"],
            "frost": ["frost", "freezing", "cold", "frosty"],
            "ice": ["ice", "icy", "frozen"],
            "cloud": ["cloud", "cloudy", "overcast"],
        },
    },
    "everyday": {
        "edges": [
            ("hunger", "ate", 0.80, +1),
            ("ate", "fullness", 0.85, +1),
            ("thirst", "drank", 0.85, +1),
            ("alarm", "woke", 0.85, +1),
            ("tired", "slept", 0.80, +1),
            ("coupon", "discount", 0.80, +1),
            ("study", "knowledge", 0.80, +1),
            ("knowledge", "passed_exam", 0.75, +1),
            ("practice", "skill", 0.80, +1),
            ("rain_outside", "umbrella", 0.70, +1),
            ("late", "rushed", 0.75, +1),
            ("noise", "distraction", 0.70, +1),
            ("distraction", "mistake", 0.65, +1),
        ],
        "alias": {
            "hunger": ["hungry", "hunger", "starving", "famished"],
            "ate": ["ate", "eat", "eating", "meal", "food"],
            "fullness": ["full", "fullness", "satisfied", "sated"],
            "thirst": ["thirst", "thirsty", "parched"],
            "drank": ["drank", "drink", "drinking", "sipped"],
            "alarm": ["alarm", "alarm clock", "buzzer"],
            "woke": ["woke", "wake", "awoke", "awakened"],
            "tired": ["tired", "sleepy", "exhausted", "weary"],
            "slept": ["slept", "sleep", "nap", "napped"],
            "coupon": ["coupon", "voucher", "promo"],
            "discount": ["discount", "deal", "savings", "cheaper"],
            "study": ["study", "studied", "studying", "revised"],
            "knowledge": ["knowledge", "learned", "learnt", "understood"],
            "passed_exam": ["passed", "pass", "aced", "exam"],
            "practice": ["practice", "practised", "rehearsed", "drilled"],
            "skill": ["skill", "skilled", "proficient", "mastery"],
            "rain_outside": ["raining", "downpour", "wet weather"],
            "umbrella": ["umbrella", "raincoat", "shelter"],
            "late": ["late", "delayed", "overslept"],
            "rushed": ["rushed", "hurried", "scrambled"],
            "noise": ["noise", "noisy", "loud", "racket"],
            "distraction": ["distraction", "distracted", "interrupted"],
            "mistake": ["mistake", "error", "slip-up", "blunder"],
        },
    },
}


def build_domain(name: str, fuzzy: bool = True) -> CausalModel:
    spec = _SPECS[name]
    m = CausalModel(domain=name, fuzzy=fuzzy)
    for c, e, s, pol in spec["edges"]:
        m.add_edge(c, e, s, pol)
    for var, aliases in spec["alias"].items():
        m.add_variable(var)
        m.add_aliases(var, aliases)
    return m


def build_all(fuzzy: bool = True) -> Dict[str, CausalModel]:
    return {name: build_domain(name, fuzzy=fuzzy) for name in _SPECS}


def domain_aliases() -> Dict[str, List[str]]:
    """Flat per-domain alias vocabulary for the router."""
    out: Dict[str, List[str]] = {}
    for name, spec in _SPECS.items():
        words: List[str] = []
        for aliases in spec["alias"].values():
            words.extend(a.lower() for a in aliases)
        out[name] = sorted(set(words))
    return out


def build_learned(path: str = None, fuzzy: bool = True):
    """Load the mined `learned` graph (causefusion.learn), or None if absent."""
    import json
    from pathlib import Path
    path = path or str(Path(__file__).resolve().parent.parent / "datasets" / "learned_edges.json")
    if not Path(path).exists():
        return None
    data = json.load(open(path, encoding="utf-8"))
    m = CausalModel(domain="learned", fuzzy=fuzzy)
    for c, e, s, pol in data["edges"]:
        m.add_edge(c, e, float(s), int(pol))
    for node in data.get("nodes", []):
        m.add_variable(node)
        m.add_aliases(node, [node, node + "s", node + "ed", node + "ing", node + "e"])
    return m


def totals():
    """(n_domains, n_nodes, n_edges) actually built."""
    models = build_all()
    nodes = sum(len(m.variables) for m in models.values())
    edges = sum(len(m.edges) for m in models.values())
    return len(models), nodes, edges
