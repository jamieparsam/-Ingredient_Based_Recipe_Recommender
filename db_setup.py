import os
import json
import pandas as pd
from dotenv import load_dotenv

# Environment (store credentials)
load_dotenv()

PG_HOST     = os.getenv("PG_HOST",     "localhost")
PG_PORT     = os.getenv("PG_PORT",     "5432")
PG_DB       = os.getenv("PG_DB",       "recipe_db")
PG_USER     = os.getenv("PG_USER",     "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "password")
MONGO_URI   = os.getenv("MONGO_URI",   "mongodb://localhost:27017/")

CLEAN_CSV   = "data/cleaned_recipes.csv"  



# PostgreSQL

def get_pg_connection():
    """
    Return a psycopg2 connection (used for control over DDL statements)
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
    only scan the relevant partition and reduces I/O.
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

    # Index on prep_time for range queries 
    cur.execute("""
        CREATE INDEX idx_recipes_prep_time
        ON recipes (prep_time_minutes);
    """)

    # Feedback log that tracks rejected recommendations 
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
    Insert cleaned recipe rows into the partitioned recipes table.
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


# MongoDB 

def get_mongo_collection():
    from pymongo import MongoClient
    client = MongoClient(MONGO_URI)
    db = client["recipe_db"]
    return db["recipes"]


def setup_mongo(collection):
    """
    Drop the existing collection and create new indexes.
      - Text index on `ingredients_text` (text search for ingredient-combination queries)
      - Ascending index on `recipe_id`
      - Ascending index on `dietary_category` (filters by diet type before running the text search)
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

    Document schema: 
      {
        recipe_id:         int,
        name:              str,
        dietary_category:  str,
        description:       str,
        ingredients_list:  [str, ...],      
        ingredients_text:  str,              ←
        num_ingredients:   int,
        prep_time_minutes: int
      }
    """
    docs = []
    for _, row in df.iterrows():
        # Convert the comma separated ingredient string into a clean list
        ing_list = [i.strip() for i in str(row["ingredients"]).split(",") if i.strip()]

        docs.append({
            "recipe_id":         int(row["recipe_id"]),
            "name":              row["name"],
            "dietary_category":  row["dietary_category"],
            "description":       row.get("description", ""),
            "ingredients_list":  ing_list,
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