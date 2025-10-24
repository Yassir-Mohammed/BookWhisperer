from functions.support_classes import ChromaConnector
from settings.vector_db_settings import OVER_WRITE_COLLECTION,DEFAULT_DB_TYPES
from utilities.paths import CHROMA_PATH



def initialize_vector_dbs():
    for db in DEFAULT_DB_TYPES:
        if db == "chroma":
            _ = ChromaConnector(
                collection_name="Default_Collection",
                db_type="chroma",
                overwrite=OVER_WRITE_COLLECTION,
                db_dir=CHROMA_PATH
            )
        else:
            print(f"The specified db ({db}) is not supported yet!")
