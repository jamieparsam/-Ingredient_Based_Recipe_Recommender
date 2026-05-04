import os
import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = os.getenv("PG_PORT",     "5432")
PG_DB       = os.getenv("PG_DB",       "recipe_db")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")

CLEAN_CSV   = "data/cleaned_recipes.csv"
MIN_SIM_THRESHOLD = 0.05   # below this → low-confidence flag


class RecipeRecommender:
    """
    Content-based recipe recommender that uses TF-IDF cosine similarity.
    """

    def __init__(self, csv_path: str = CLEAN_CSV):
        self.df = pd.read_csv(csv_path)
        self.vectorizer   = None
        self.tfidf_matrix = None
        self._build_index()

  
    # Model Building

    def _build_index(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            stop_words="english",
            analyzer="word",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.df["ingredients"].fillna("")
        )
        print(f"[recommender] Index built — "
              f"{self.tfidf_matrix.shape[0]} recipes × "
              f"{self.tfidf_matrix.shape[1]} terms.")

    
    # Core Recommendation
    def recommend(
        self,
        user_ingredients: list[str],
        max_prep_time: int = 60,
        dietary_preference: str = None,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Return the top-N recipe recommendations for a student.
        """
        if not user_ingredients:
            raise ValueError("user_ingredients must not be empty.")

        # Create filters for recipes that match
        mask = self.df["prep_time_minutes"] <= max_prep_time
        if dietary_preference:
            mask &= self.df["dietary_category"] == dietary_preference

        candidates = self.df[mask].copy()
        if candidates.empty:
            print("[recommender] No candidates match the hard filters.")
            return pd.DataFrame()

        candidate_idx = candidates.index.tolist()

        # Vectorise user's ingredient query
        user_query = ", ".join([ing.lower().strip() for ing in user_ingredients])
        user_vec = self.vectorizer.transform([user_query])

        # Cosine similarity against candidate recipes only 
        candidate_matrix = self.tfidf_matrix[candidate_idx]
        sim_scores = cosine_similarity(user_vec, candidate_matrix).flatten()

        # Rank and select top-N 
        top_indices = np.argsort(sim_scores)[::-1][:top_n]

        results = candidates.iloc[top_indices].copy()
        results["similarity_score"] = sim_scores[top_indices].round(4)
        results["low_confidence"]   = results["similarity_score"] < MIN_SIM_THRESHOLD

        return results[["recipe_id", "name", "dietary_category",
                         "prep_time_minutes", "similarity_score",
                         "ingredients", "low_confidence"]].reset_index(drop=True)

    # Feedback Loop
    def log_feedback(
        self,
        session_id: str,
        recipe_id: int,
        action: str = "rejected",
    ) -> bool:
        """
        Log user feedback (accepted/rejected to the PostgreSQL feedback_log.)
          - Which recipes are rejected the most?
          - Are certain dietary categories disproportionately rejected?
          - Does rejection rate correlate with low similarity scores?
        """
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=PG_HOST, port=PG_PORT, dbname=PG_DB,
                user=PG_USER, password=PG_PASSWORD
            )
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO feedback_log (session_id, recipe_id, action, logged_at)
                VALUES (%s, %s, %s, %s);
                """,
                (session_id, recipe_id, action, datetime.datetime.utcnow())
            )
            conn.commit()
            cur.close()
            conn.close()
            print(f"[recommender] Feedback logged — recipe_id={recipe_id}, "
                  f"action={action}.")
            return True
        except Exception as exc:
            print(f"[recommender] WARNING: Could not log feedback — {exc}")
            return False

    def log_feedback_file(
        self,
        session_id: str,
        recipe_id: int,
        action: str = "rejected",
        log_path: str = "data/feedback_log.csv",
    ):
        """
        Fallback feedback logger that writes to a CSV file.
        """
        import csv
        from pathlib import Path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        file_exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["session_id", "recipe_id", "action", "logged_at"])
            writer.writerow([
                session_id, recipe_id, action,
                datetime.datetime.utcnow().isoformat()
            ])
        print(f"[recommender] Feedback written to {log_path}.")


# Demo

def demo():
    rec = RecipeRecommender()

    print("  Recipe Recommendation Demo")

    # Scenario A — vegan meal with pantry basics
    print("\nScenario A — Vegan student, 20-min limit")
    print("  Ingredients: rice, garlic, soy sauce, tofu, frozen peas")
    results_a = rec.recommend(
        user_ingredients=["rice", "garlic", "soy sauce", "tofu", "frozen peas"],
        max_prep_time=20,
        dietary_preference="Vegan",
        top_n=3,
    )
    print(results_a[["name", "prep_time_minutes", "similarity_score"]].to_string())

    # Scenario B — non-veg student with chicken
    print("\nScenario B — Non-Vegetarian, 30-min limit")
    print("  Ingredients: chicken breast, garlic, olive oil, pasta")
    results_b = rec.recommend(
        user_ingredients=["chicken breast", "garlic", "olive oil", "pasta"],
        max_prep_time=30,
        dietary_preference="Non-Vegetarian",
        top_n=3,
    )
    print(results_b[["name", "prep_time_minutes", "similarity_score"]].to_string())

    # Simulate a rejection 
    if not results_a.empty:
        rec.log_feedback_file(
            session_id="demo_session_001",
            recipe_id=int(results_a.iloc[0]["recipe_id"]),
            action="rejected",
        )


if __name__ == "__main__":
    demo()
