"""
db_setup.py
===========
Initialises both databases and loads cleaned data into them.

We split storage across two database:

  PostgreSQL (relational)
  -----------------------
  Stores the *structured* recipe metadata like recipe_id, name, prep_time,
  dietary_category, cuisine_type. These fields have well-defined types,
  support efficient range queries ("all recipes under 20 min"), and benefit
  from SQL JOINs with the feedback_log table.

  Table partitioning is applied on dietary_category so that a query like
  "show me all Vegan recipes" hits only the Vegan partition — important
  when the table grows to millions of rows.

  MongoDB (document store)
  ------------------------
  Stores the *flexible* ingredient list and free-text description for each
  recipe. Ingredient lists vary in length and format (arrays vs.
  comma strings) so MongoDB's schema-less documents adapt to this.
  We create a text index on the `ingredients` field to support fast
  ingredient combination searches without a full collection scan.

Usage:
  # Ensure .env has PG_* and MONGO_URI set (see README).
  python db_setup.py
"""

import os
import json
import pandas as pd
from dotenv import load_dotenv

# Environment 
# We store credentials in a .env file so they are never hard-coded.
load_dotenv()

PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = os.getenv("PG_PORT",     "5432")
PG_DB       = os.getenv("PG_DB",       "recipe_db")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")
MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://localhost:27017/")

CLEAN_CSV   = "data/cleaned_recipes.csv"   # output of etl_pipeline.py



# PostgreSQL helpers

def get_pg_connection():
    """
    Return a raw psycopg2 connection.

    We use psycopg2 directly for explicit control
    over DDL statements like CREATE TABLE … PARTITION BY, which some ORM
    layers abstract away in ways that make partitioning harder to see.
    """
    import psycopg2
    return psycopg2.connect(
        host=PG_HOST, port=PG_PORT, dbname=PG_DB,
        user=PG_USER, password=PG_PASSWORD
    )


def setup_postgres(conn):
    """
    Create the partitioned recipes table and the feedback_log table.

    LIST partitioning on dietary_category.
    Each partition (Vegan, Vegetarian, Non-Vegetarian) is stored as a
    separate physical table segment so queries filtered by diet type
    only scan the relevant partition, reducing I/O substantially.
    """
    cur = conn.cursor()

    # Drop & recreate for a clean slate 
    cur.execute("DROP TABLE IF EXISTS feedback_log CASCADE;")
    cur.execute("DROP TABLE IF EXISTS recipes CASCADE;")

    # Parent partitioned table 
    cur.execute("""
        CREATE TABLE recipes (
            recipe_id        SERIAL,
            name             VARCHAR(255)  NOT NULL,
            prep_time_minutes INT           NOT NULL,
            dietary_category  VARCHAR(50)   NOT NULL,
            cuisine_type      VARCHAR(50),
            source            VARCHAR(50)   DEFAULT 'synthetic',
            PRIMARY KEY (recipe_id, dietary_category)   -- partition key must be in PK
        ) PARTITION BY LIST (dietary_category);
    """)

    # One physical partition per dietary category 
    for category in ("Vegan", "Vegetarian", "Non-Vegetarian"):
        safe = category.replace("-", "_").replace(" ", "_")
        cur.execute(f"""
            CREATE TABLE recipes_{safe}
            PARTITION OF recipes
            FOR VALUES IN ('{category}');
        """)
        print(f"[db_setup] Created partition: recipes_{safe}")

    # Index on prep_time for range queries ("recipes under 20 min") 
    cur.execute("""
        CREATE INDEX idx_recipes_prep_time
        ON recipes (prep_time_minutes);
    """)

    # Feedback log: tracks rejected recommendations 
    cur.execute("""
        CREATE TABLE feedback_log (
            log_id        SERIAL PRIMARY KEY,
            session_id    VARCHAR(64),
            recipe_id     INT,
            action        VARCHAR(20) CHECK (action IN ('rejected', 'accepted')),
            logged_at     TIMESTAMP DEFAULT NOW()
        );
    """)

    conn.commit()
    cur.close()
    print("[db_setup] PostgreSQL schema created successfully.")


def load_postgres(conn, df: pd.DataFrame):
    """
    Bulk-insert cleaned recipe rows into the partitioned recipes table.

    We use executemany rather than COPY because the dataset is small (<5k rows).
    For production scale, switch to psycopg2's copy_from() or the
    COPY FROM STDIN protocol for orders-of-magnitude faster bulk loads.
    """
    cur = conn.cursor()

    pg_cols = ["recipe_id", "name", "prep_time_minutes",
               "dietary_category", "cuisine_type", "source"]
    rows = df[pg_cols].values.tolist()

    cur.executemany(
        """
        INSERT INTO recipes
            (recipe_id, name, prep_time_minutes, dietary_category, cuisine_type, source)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """,
        rows
    )
    conn.commit()
    cur.close()
    print(f"[db_setup] Inserted {len(rows)} rows into PostgreSQL.")


# MongoDB helpers

def get_mongo_collection():
    """
    Return the pymongo Collection for recipe documents.

    MongoDB is chosen for the ingredient data because ingredient lists are
    variable-length arrays. The full text search over ingredient names is
    supported by MongoDB's $text index that tokenises ingredient strings. The
    recipe descriptions and user notes can also be added to each document
    without a schema migration.
    """
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    db = client["recipe_db"]
    return db["recipes"]


def setup_mongo(collection):
    """
    Drop the existing collection and create new indexes.
      • Text index on `ingredients_text`: enables $text search for
        ingredient-combination queries ("find recipes with garlic and rice").
      • Ascending index on `recipe_id`: foreign-key lookup from PostgreSQL.
      • Ascending index on `dietary_category`: filter documents by diet type
        before running the text search, reducing the search space.
    """
    from pymongo import TEXT, ASCENDING

    collection.drop()

    collection.create_index(
        [("ingredients_text", TEXT)],
        name="idx_ingredients_text",
        default_language="english"
    )
    collection.create_index([("recipe_id", ASCENDING)],
                            name="idx_recipe_id", unique=True)
    collection.create_index([("dietary_category", ASCENDING)],
                            name="idx_dietary_category")

    print("[db_setup] MongoDB indexes created.")


def load_mongo(collection, df: pd.DataFrame):
    """
    Insert one document per recipe into MongoDB.

    Document schema (flexible — can be extended at runtime):
      {
        recipe_id:         int,
        name:              str,
        dietary_category:  str,
        description:       str,
        ingredients_list:  [str, ...],       ← array for easy $in queries
        ingredients_text:  str,              ← concatenated for $text index
        num_ingredients:   int,
        prep_time_minutes: int
      }
    """
    docs = []
    for _, row in df.iterrows():
        # Convert the comma-separated ingredient string into a clean list
        ing_list = [i.strip() for i in str(row["ingredients"]).split(",") if i.strip()]

        docs.append({
            "recipe_id":         int(row["recipe_id"]),
            "name":              row["name"],
            "dietary_category":  row["dietary_category"],
            "description":       row.get("description", ""),
            "ingredients_list":  ing_list,
            # Concatenated text field enables the $text index to tokenise
            # each ingredient name independently for fast full-text search.
            "ingredients_text":  " ".join(ing_list),
            "num_ingredients":   len(ing_list),
            "prep_time_minutes": int(row["prep_time_minutes"]),
        })

    if docs:
        collection.insert_many(docs, ordered=False)
    print(f"[db_setup] Inserted {len(docs)} documents into MongoDB.")


# Execute

def main():
    # Load the ETL-cleaned dataset
    if not os.path.exists(CLEAN_CSV):
        raise FileNotFoundError(
            f"{CLEAN_CSV} not found. Run etl_pipeline.py first."
        )
    df = pd.read_csv(CLEAN_CSV)
    print(f"[db_setup] Loaded {len(df)} cleaned recipes from {CLEAN_CSV}.")

    # PostgreSQL 
    pg_conn = get_pg_connection()
    setup_postgres(pg_conn)
    load_postgres(pg_conn, df)
    pg_conn.close()

    # MongoDB 
    mongo_col = get_mongo_collection()
    setup_mongo(mongo_col)
    load_mongo(mongo_col, df)

    print("[db_setup] Both databases populated successfully.")


if __name__ == "__main__":
    main()
