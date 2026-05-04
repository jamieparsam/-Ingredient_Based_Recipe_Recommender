import sys
import pandas as pd
import numpy as np
from recommender import RecipeRecommender

# Simulated student scenarios
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
    Find the relevant recipe IDs for a given student query.
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
    Precision@K is the fraction of recommended items that are relevant
    """
    if not recommended_ids:
        return 0.0
    hits = sum(1 for rid in recommended_ids if rid in relevant_ids)
    return hits / len(recommended_ids)


def recall_at_k(recommended_ids: list[int], relevant_ids: set[int]) -> float:
    """
    Recall@K is the fraction of all relevant items that were recommended.
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
    Measure whether the model is biased torwards one dietary category.
    """
    # Neutral query pool that have ingredients stored in all dietary categories
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
            max_prep_time=120,     
            dietary_preference=None,
            top_n=20,             
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


# Main Evaluation

def run_evaluation():
   
    rec = RecipeRecommender()
    df = rec.df

    scenario_results = []

    for sc in SCENARIOS:
        print(f"\n")
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

        print(f"\n  Relevant recipes : {len(relevant_ids)}")
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
    print(f"\n")
    print("  Aggregate Metrics With All Scenarios")
    print(f"\n")
    res_df = pd.DataFrame(scenario_results)
    print(res_df[["scenario", "label", "precision", "recall", "f1"]].to_string(index=False))

    avg_p = res_df["precision"].mean()
    avg_r = res_df["recall"].mean()
    avg_f = res_df["f1"].mean()
    print(f"\n  Mean Precision : {avg_p:.4f}")
    print(f"  Mean Recall    : {avg_r:.4f}")
    print(f"  Mean F1        : {avg_f:.4f}")

    # Bias analysis
    print(f"\n")
    print("Bias Analysis")
    print(f"\n")
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
        print("\n  All pairwise biases within acceptable threshold (|bias| < 0.05).")
    else:
        print("\n  ⚠ Bias detected. Consider re-weighting ingredients by dietary category.")

    return res_df, bias


if __name__ == "__main__":
    run_evaluation()
