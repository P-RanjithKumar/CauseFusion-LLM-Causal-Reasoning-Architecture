"""Sanity check: compare Jaccard vs MiniLM alignment."""
from sentence_transformers import SentenceTransformer
from utils.evaluation import evaluate_explanation_alignment

print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
st = SentenceTransformer("all-MiniLM-L6-v2")

# Simulate SCM paths and human explanations (e-CARE style)
paths = ["rain→wetness", "force→breaking", "threat→fear", "none"]
expls = [
    "heavy precipitation poured from the sky, making everything damp",
    "the strong impact shattered the glass into pieces",
    "the dangerous situation made him feel terrified",
    "something happened",
]

print("\n--- Running Explanation Alignment Scoring ---")

# Old: Jaccard
old = evaluate_explanation_alignment(paths, expls, st_model=None)
print("Jaccard mean alignment:  {:.4f}".format(old["mean_alignment"]))
print("Jaccard individual:     {}".format([round(s, 4) for s in old["per_item_scores"]]))

# New: MiniLM dense
new = evaluate_explanation_alignment(paths, expls, st_model=st)
print("MiniLM mean alignment:   {:.4f}".format(new["mean_alignment"]))
print("MiniLM individual:      {}".format([round(s, 4) for s in new["per_item_scores"]]))

improvement = new["mean_alignment"] / max(old["mean_alignment"], 0.0001)
print("\nImprovement: {:.1f}x".format(improvement))
