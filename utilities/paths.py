import os
from settings.vector_db_settings import DEFAULT_DB_TYPES


# ========================
# TEMP DATA DIRECTORIES
# ========================
MAIN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TEMP_DIR = os.path.join(MAIN_DIR, "temp")
RAW_DATA_DIR = os.path.join(TEMP_DIR, "1_raw_data")
PARSED_DATA_DIR = os.path.join(TEMP_DIR, "2_parsed_data")
SPLIT_DATA_DIR = os.path.join(TEMP_DIR, "3_cleaned_data")

# Create temp directories if they don't exist
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PARSED_DATA_DIR, exist_ok=True)
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

# ========================
# VECTOR DATABASE DIRECTORIES
# ========================
VECTORDB_DIR = os.path.join(MAIN_DIR, "vector_database")
os.makedirs(VECTORDB_DIR, exist_ok=True)

# Create subfolders for each DB type
for db in DEFAULT_DB_TYPES:
    if db not in ["chroma", "leann", "Qdrant", "Weaviate", "pgvector"]:
        raise ValueError(f"Invalid DB type '{db}'. Please use one of {DEFAULT_DB_TYPES}.")
    db_path = os.path.join(VECTORDB_DIR, db)
    os.makedirs(db_path, exist_ok=True)
    # Create a dynamic variable like CHROMA_PATH, QDRANT_PATH, etc.
    globals()[f"{db.upper()}_PATH"] = db_path
