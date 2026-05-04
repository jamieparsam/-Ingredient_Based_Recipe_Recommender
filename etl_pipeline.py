import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# Paths
SYNTHETIC_CSV  = "data/synthetic_recipes.csv"
KAGGLE_A_CSV   = "data/raw/food_cuisines.csv"
KAGGLE_B_CSV   = "data/raw/food_recipes.csv"
OUTPUT_CSV     = "data/cleaned_recipes.csv"


# Extract

def load_synthetic(path: str = SYNTHETIC_CSV) -> pd.DataFrame:
    """Load synthetic recipes produced by data_gen.py."""
    df = pd.read_csv(path)
    print(f"[etl] Loaded {len(df)} synthetic recipes from {path}.")
    return df


def load_kaggle_a(path: str = KAGGLE_A_CSV) -> pd.DataFrame | None:
    """
    Load the Kaggle food dataset (Food_ID, Name, C_Type, Veg_Non, Describe).

      Name       → name
      Veg_Non    → dietary_category  (veg -> Vegetarian, non-veg -> Non-Vegetarian)
      Describe   → ingredients  
      C_Type     → cuisine_type
    """
    if not os.path.exists(path):
        print(f"[etl] Kaggle-A not found at {path} — skipping.")
        return None

    df = pd.read_csv(path)

    # Map Veg_Non values to the dietary_category labels
    diet_map = {"veg": "Vegetarian", "non-veg": "Non-Vegetarian"}
    df["dietary_category"] = df["Veg_Non"].str.strip().map(diet_map).fillna("Vegetarian")

    df = df.rename(columns={
        "Food_ID":  "recipe_id",
        "Name":     "name",
        "C_Type":   "cuisine_type",
        "Describe": "ingredients",
    })

    # Kaggle-A has no prep_time so we will fill it with category mean after merge
    df["prep_time_minutes"] = np.nan
    df["description"]       = ""
    df["source"]            = "kaggle_a"
    df["num_ingredients"]   = df["ingredients"].apply(
        lambda x: len(str(x).split(","))
    )

    keep = ["recipe_id", "name", "prep_time_minutes", "dietary_category",
            "cuisine_type", "description", "ingredients", "num_ingredients", "source"]
    print(f"[etl] Loaded {len(df)} recipes from Kaggle-A.")
    return df[keep]


def load_kaggle_b(path: str = KAGGLE_B_CSV) -> pd.DataFrame | None:
    """
    Load the other Kaggle dataset

      recipe_title -> name
      diet         -> dietary_category 
      prep_time    -> prep_time_minutes 
      ingredients  -> ingredients 
      description  -> description
      cuisine      -> cuisine_type
    """
    if not os.path.exists(path):
        print(f"[etl] Kaggle-B not found at {path} — skipping.")
        return None

    # Read only the columns needed to save memory 
    usecols = ["recipe_title", "diet", "prep_time", "ingredients",
               "description", "cuisine"]
    df = pd.read_csv(path, usecols=usecols, nrows=5000)

    # 'diet' column normalisation 
    def normalise_diet(val: str) -> str:
        v = str(val).lower()
        if "vegan" in v:
            return "Vegan"
        if "non" in v or "meat" in v or "chicken" in v or "fish" in v:
            return "Non-Vegetarian"
        return "Vegetarian"

    df["dietary_category"] = df["diet"].apply(normalise_diet)

    # Prep time parsing 
    # Take the numeric portion and convert to integer minutes.
    def parse_prep(val) -> float:
        if pd.isna(val):
            return np.nan
        numbers = re.findall(r"\d+", str(val))
        if not numbers:
            return np.nan
        mins = int(numbers[-1])  # last number is usually minutes
        # if first token contains "hour" the leading number is hours
        if "hour" in str(val).lower() and len(numbers) >= 1:
            mins = int(numbers[0]) * 60 + (int(numbers[1]) if len(numbers) > 1 else 0)
        return float(mins)

    df["prep_time_minutes"] = df["prep_time"].apply(parse_prep)

    # Ingredient format normalisation 
    # Kaggle-B uses (|) as a separator so we will convert it to comma
    df["ingredients"] = df["ingredients"].str.replace("|", ", ", regex=False)

    df = df.rename(columns={
        "recipe_title": "name",
        "cuisine":      "cuisine_type",
    })
    df["source"]          = "kaggle_b"
    df["num_ingredients"] = df["ingredients"].apply(
        lambda x: len(str(x).split(","))
    )

    # Assign recipe_ids
    df["recipe_id"] = range(10001, 10001 + len(df))

    keep = ["recipe_id", "name", "prep_time_minutes", "dietary_category",
            "cuisine_type", "description", "ingredients", "num_ingredients", "source"]
    print(f"[etl] Loaded {len(df)} recipes from Kaggle-B.")
    return df[keep]


# Transform

def _clean_ingredient_string(ing_str: str) -> str:
    """
    Standardise comma-separated ingredient string.
    - Lowercase everything ex "Olive Oil" and "olive oil" are the same.
    - Strip leading or trailing whitespace from each token.
    - Remove quantity tokens that are numeric or measurement words ex "2 tbsp" or "1/4 cup"
      so the ML model matches on ingredient names, not quantities. Quantities are stored
      separately in MongoDB.
    - Deduplicate within the same recipe ingredient list.
    - Sort alphabetically so that TF-IDF vector order is predicable.
    """
    MEASURE_RE = re.compile(
        r"^\d[\d/.]*\s*(tbsp|tsp|cup|oz|g|kg|ml|l|lb|pinch|bunch|clove|"
        r"slice|piece|can|pkg|package|to\s+taste)?$",
        re.IGNORECASE
    )

    tokens = [t.strip().lower() for t in str(ing_str).split(",")]
    cleaned = []
    for tok in tokens:
        if not tok:
            continue
        if MEASURE_RE.match(tok):
            continue   
        cleaned.append(tok)

    # Deduplicate while keeping order 
    seen = dict.fromkeys(cleaned)
    return ", ".join(sorted(seen.keys()))


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Some of the things to keep in mind to clean combined dataframe:

    1. Drop duplicates
       The same duplicate rows add no information and increase model confidence.

    2. Deduplicate on (name, ingredients)
       Two recipes with the same name AND ingredients are the same dish even if 
       other information about the dish is different.

    3. Standardise ingredient strings
       Assign to _clean_ingredient_string().

    4. Standardise text columns
       Clear whitespace and title-case recipe names for visual consistency.

    5. Fill in missing prep_time_minutes for dishes
       Fill with the *mean prep time of the same dietary_category*.
       Vegan meals tend to be quicker than Non-Vegetarian ones. Using a category-specific
       mean avoids biasing short-prep recipes.

    6. Look for outliers
       Prep times > 120 min are likely data errors in the Kaggle dataset. We limit prep
       times to 120 to keep the prediction model relevant to college students.

    7. Reassign sequential recipe_ids
       After merging multiple sources the recipe_id column may have gaps
       or conflicts. We reassign a clean integer index.
    """
    print(f"[etl] Starting clean — {len(df)} rows.")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[etl]   Dropped {before - len(df)} exact duplicate rows.")

    # Deduplicate on name + ingredients
    before = len(df)
    df["_dedup_key"] = (
        df["name"].str.lower().str.strip() + "||" +
        df["ingredients"].str.lower().str.strip()
    )
    df = df.drop_duplicates(subset="_dedup_key").drop(columns="_dedup_key")
    print(f"[etl]   Dropped {before - len(df)} name+ingredient duplicates.")

    # Clean ingredient strings
    df["ingredients"] = df["ingredients"].apply(_clean_ingredient_string)

    # Standardise text columns
    df["name"]             = df["name"].str.strip().str.title()
    df["dietary_category"] = df["dietary_category"].str.strip()
    df["cuisine_type"]     = df["cuisine_type"].fillna("Unknown").str.strip().str.title()
    df["description"]      = df["description"].fillna("").str.strip()

    # Fill missing prep_time with category mean
    cat_mean = df.groupby("dietary_category")["prep_time_minutes"].transform("mean")
    missing_mask = df["prep_time_minutes"].isna()
    df.loc[missing_mask, "prep_time_minutes"] = cat_mean[missing_mask].round()
    # Fallback for any remaining NaN (e.g. a category with all-NaN times)
    df["prep_time_minutes"] = df["prep_time_minutes"].fillna(20).astype(int)
    print(f"[etl]   Filled {missing_mask.sum()} missing prep_time values.")

    # Limit and get rid of longer prep times
    before_clamp = (df["prep_time_minutes"] > 120).sum()
    df["prep_time_minutes"] = df["prep_time_minutes"].clip(upper=120)
    if before_clamp:
        print(f"[etl]   Clamped {before_clamp} rows with prep_time > 120 min.")

    # Reindex recipe_id
    df = df.reset_index(drop=True)
    df["recipe_id"] = df.index + 1

    # Update num_ingredients to reflect cleaned lists
    df["num_ingredients"] = df["ingredients"].apply(
        lambda x: len([t for t in x.split(",") if t.strip()])
    )

    print(f"[etl] Clean complete — {len(df)} rows remain.")
    return df



# Load to CSV 

def save_cleaned(df: pd.DataFrame, path: str = OUTPUT_CSV):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[etl] Cleaned dataset saved to {path} ({len(df)} rows).")



def run_pipeline(
    use_kaggle_a: bool = True,
    use_kaggle_b: bool = True,
) -> pd.DataFrame:
    frames = [load_synthetic()]

    if use_kaggle_a:
        ka = load_kaggle_a()
        if ka is not None:
            frames.append(ka)

    if use_kaggle_b:
        kb = load_kaggle_b()
        if kb is not None:
            frames.append(kb)

    # Merge all sources 
    df = pd.concat(frames, ignore_index=True)
    print(f"[etl] Combined {len(df)} rows from {len(frames)} source(s).")

    df = clean_dataframe(df)
    save_cleaned(df)
    return df


if __name__ == "__main__":
    run_pipeline(use_kaggle_a=True, use_kaggle_b=True)
