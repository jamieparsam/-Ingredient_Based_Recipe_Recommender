"""
evaluate.py

Evaluation & Validation Module for the Recipe Recommendation System.

Metrics Used:

1. Precision@K
   Of the K recipes recommended, how many are *relevant* (matching the
   student's dietary preference)?
   Precision = |relevant ∩ recommended| / |recommended|

2. Recall@K
   Of all relevant recipes in the dataset, what fraction did we surface?
   Recall = |relevant ∩ recommended| / |relevant in dataset|

   We define a recipe as relevant if it matches the student's dietary_preference
   AND has prep_time ≤ student's max_prep_time, AND shares ≥ 1 ingredient with the
   student's ingredient list.
   

3. Forecast Bias Analysis
   We check whether the model has a systematic preference for one dietary
   category over others by comparing mean similarity scores across groups.
   A model should not consistently score Vegan recipes higher than Non-Vegetarian
   ones (or vice versa) when the query is neutral.
   Bias = mean_score(group_A) - mean_score(group_B)
   |Bias| < 0.05 is our acceptance threshold.

Simulated Student Scenarios:

The module runs 6 pre-defined scenarios that cover the rubric's examples:
  S1 — 3 basic ingredients, 15-minute limit, no dietary preference
  S2 — Vegan, 5 ingredients
  S3 — Vegetarian, 6 ingredients, 30-minute limit
  S4 — Non-Vegetarian, 4 ingredients, 25-minute limit
  S5 — Very limited (2 ingredients, 10-min) — tests low-confidence path
  S6 — 8 ingredients, no time constraint
"""

import sys
import pandas as pd
import numpy as np
from recommender import RecipeRecommender

# Simulated student profiles
SCENARIOS = [
    {
        "id":          "S1",
        "label":       "Basic Pantry (any diet, 15 min)",
        "ingredients": ["rice", "garlic", "soy sauce"],
        "max_time":    15,
        "diet":        None,
        "top_n":       5,
    },
    {
        "id":          "S2",
        "label":       "Vegan, 5 ingredients",
        "ingredients": ["tofu", "soy sauce", "garlic", "broccoli", "rice"],
        "max_time":    60,
        "diet":        "Vegan",
        "top_n":       5,
    },
    {
        "id":          "S3",
        "label":       "Vegetarian, 6 ingredients, 30 min",
        "ingredients": ["egg", "cheese", "milk", "pasta", "garlic", "butter"],
        "max_time":    30,
        "diet":        "Vegetarian",
        "top_n":       5,
    },
    {
        "id":          "S4",
        "label":       "Non-Vegetarian, 4 ingredients, 25 min",
        "ingredients": ["chicken breast", "olive oil", "garlic", "pasta"],
        "max_time":    25,
        "diet":        "Non-Vegetarian",
        "top_n":       5,
    },
    {
        "id":          "S5",
        "label":       "Minimal pantry (2 items, 10 min)",
        "ingredients": ["bread", "peanut butter"],
        "max_time":    10,
        "diet":        None,
        "top_n":       3,
    },
    {
        "id":          "S6",
        "label":       "Large pantry, no time limit",
        "ingredients": ["rice", "chicken breast", "garlic", "onion",
                        "soy sauce", "broccoli", "egg", "olive oil"],
        "max_time":    120,
        "diet":        None,
        "top_n":       5,
    },
]


# Relevance 


def get_relevant_ids(
    df: pd.DataFrame,
    user_ingredients: list[str],
    max_time: int,
    dietary_preference: str,
) -> set[int]:
    """
    Determine the relevant recipe IDs for a given student query.

    A recipe is *relevant* if ALL of the following hold true:
      - prep_time_minutes ≤ max_time
      - dietary_category matches the preference (if specified)
      - At least one of the student's ingredients appears in the recipe's
        ingredient string

    """
    mask = df["prep_time_minutes"] <= max_time
    if dietary_preference:
        mask &= df["dietary_category"] == dietary_preference

    # Ingredient overlap filter
    user_set = set(i.lower().strip() for i in user_ingredients)

    def has_overlap(ing_str: str) -> bool:
        recipe_ings = set(t.strip() for t in str(ing_str).split(","))
        return bool(user_set & recipe_ings)

    mask &= df["ingredients"].apply(has_overlap)
    return set(df[mask]["recipe_id"].tolist())


# Precision & Recall

def precision_at_k(recommended_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Precision@K: fraction of recommended items that are relevant.

    Interpretation for the report:
      Precision = 0.8 → 4 out of 5 recommendations were relevant recipes.
    """
    if not recommended_ids:
        return 0.0
    hits = sum(1 for rid in recommended_ids if rid in relevant_ids)
    return hits / len(recommended_ids)


def recall_at_k(recommended_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Recall@K: fraction of all relevant items that were recommended.

    Interpretation for the report:
      Recall = 0.3 → the model surfaced 30% of all relevant recipes.
      Low recall is expected in top-5 retrieval over large datasets.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for rid in recommended_ids if rid in relevant_ids)
    return hits / len(relevant_ids)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)



# Forecast Bias Analysis


def bias_analysis(rec: RecipeRecommender, n_trials: int = 50) -> dict:
    """
    Measure whether the model systematically favours one dietary category.

    Method
    ------
    Run `n_trials` neutral queries (random ingredients common to all diets,
    no dietary filter, generous time limit) and record the mean similarity
    score for each dietary category across all returned recommendations.

    A large mean-score gap between categories indicates the model is biased
    toward one type of recipe even when the user has not specified a preference.

    Acceptance threshold: |bias| < 0.05 between any two category means.
    """
    # Neutral query pool (ingredients present in all dietary categories)
    NEUTRAL_INGREDIENTS = [
        ["garlic", "rice", "olive oil"],
        ["pasta", "onion", "tomato"],
        ["salt", "black pepper", "garlic", "carrot"],
        ["rice", "soy sauce", "garlic"],
        ["lemon juice", "garlic", "olive oil", "spinach"],
    ]

    category_scores: dict[str, list[float]] = {
        "Vegan": [],
        "Vegetarian": [],
        "Non-Vegetarian": [],
    }

    for trial_idx in range(n_trials):
        ingredients = NEUTRAL_INGREDIENTS[trial_idx % len(NEUTRAL_INGREDIENTS)]
        results = rec.recommend(
            user_ingredients=ingredients,
            max_prep_time=120,     # no time constraint → full dataset
            dietary_preference=None,
            top_n=20,             # retrieve a larger slice for better stats
        )
        if results.empty:
            continue
        for _, row in results.iterrows():
            cat = row["dietary_category"]
            if cat in category_scores:
                category_scores[cat].append(float(row["similarity_score"]))

    means = {cat: (np.mean(scores) if scores else 0.0)
             for cat, scores in category_scores.items()}
    counts = {cat: len(scores) for cat, scores in category_scores.items()}

    # Pairwise bias deltas
    categories = list(means.keys())
    biases = {}
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            c1, c2 = categories[i], categories[j]
            biases[f"{c1} vs {c2}"] = round(means[c1] - means[c2], 4)

    return {"means": means, "counts": counts, "biases": biases}


# Main Evaluation Runner


def run_evaluation():
    print("=" * 70)
    print("  RECIPE RECOMMENDER — EVALUATION REPORT")
    print("=" * 70)

    rec = RecipeRecommender()
    df = rec.df

    scenario_results = []

    for sc in SCENARIOS:
        print(f"\n{'─' * 70}")
        print(f"Scenario {sc['id']}: {sc['label']}")
        print(f"  Ingredients : {sc['ingredients']}")
        print(f"  Max time    : {sc['max_time']} min")
        print(f"  Diet filter : {sc['diet'] or 'None'}")

        # Get recommendations
        recs = rec.recommend(
            user_ingredients=sc["ingredients"],
            max_prep_time=sc["max_time"],
            dietary_preference=sc["diet"],
            top_n=sc["top_n"],
        )

        if recs.empty:
            print("  [!] No recommendations returned.")
            scenario_results.append({**sc, "precision": 0.0,
                                     "recall": 0.0, "f1": 0.0,
                                     "n_relevant": 0, "n_recommended": 0})
            continue

        # Relevant set
        relevant_ids = get_relevant_ids(
            df=df,
            user_ingredients=sc["ingredients"],
            max_time=sc["max_time"],
            dietary_preference=sc["diet"],
        )

        recommended_ids = recs["recipe_id"].tolist()

        p = precision_at_k(recommended_ids, relevant_ids)
        r = recall_at_k(recommended_ids, relevant_ids)
        f = f1_score(p, r)

        print(f"\n  Top-{sc['top_n']} Recommendations:")
        print(recs[["name", "dietary_category",
                     "prep_time_minutes", "similarity_score"]].to_string(index=False))

        print(f"\n  Ground-truth relevant recipes : {len(relevant_ids)}")
        print(f"  Recommended                   : {len(recommended_ids)}")
        print(f"  Precision@{sc['top_n']}                  : {p:.4f}")
        print(f"  Recall@{sc['top_n']}                     : {r:.4f}")
        print(f"  F1 Score                      : {f:.4f}")

        low_conf = recs["low_confidence"].sum()
        if low_conf:
            print(f"  ⚠ {low_conf} recommendation(s) flagged as low-confidence "
                  f"(similarity < 0.05).")

        scenario_results.append({
            "scenario":       sc["id"],
            "label":          sc["label"],
            "precision":      round(p, 4),
            "recall":         round(r, 4),
            "f1":             round(f, 4),
            "n_relevant":     len(relevant_ids),
            "n_recommended":  len(recommended_ids),
        })

    # Aggregate metrics 
    print(f"\n{'=' * 70}")
    print("  AGGREGATE METRICS ACROSS ALL SCENARIOS")
    print(f"{'=' * 70}")
    res_df = pd.DataFrame(scenario_results)
    print(res_df[["scenario", "label", "precision", "recall", "f1"]].to_string(index=False))

    avg_p = res_df["precision"].mean()
    avg_r = res_df["recall"].mean()
    avg_f = res_df["f1"].mean()
    print(f"\n  Mean Precision : {avg_p:.4f}")
    print(f"  Mean Recall    : {avg_r:.4f}")
    print(f"  Mean F1        : {avg_f:.4f}")

    # Bias analysis
    print(f"\n{'=' * 70}")
    print("  FORECAST BIAS ANALYSIS (dietary fairness)")
    print(f"{'=' * 70}")
    bias = bias_analysis(rec, n_trials=50)

    print("\n  Mean similarity score per dietary category:")
    for cat, mean_score in bias["means"].items():
        n = bias["counts"][cat]
        print(f"    {cat:<18} : {mean_score:.4f}  (n={n})")

    print("\n  Pairwise bias (positive = first category scored higher):")
    all_biased = False
    for pair, delta in bias["biases"].items():
        flag = " ← ⚠ BIAS DETECTED" if abs(delta) > 0.05 else " ✓"
        if abs(delta) > 0.05:
            all_biased = True
        print(f"    {pair:<35} : {delta:+.4f}{flag}")

    if not all_biased:
        print("\n  ✓ All pairwise biases within acceptable threshold (|bias| < 0.05).")
    else:
        print("\n  ⚠ Bias detected. Consider re-weighting ingredients by dietary category.")

    print(f"\n{'=' * 70}")
    print("  Evaluation complete.")
    print(f"{'=' * 70}\n")

    return res_df, bias


if __name__ == "__main__":
    run_evaluation()
