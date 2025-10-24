import streamlit as st
from functions.GUI import upload_files_element
from utilities.vectordb_initialization import initialize_vector_dbs
from functions.support_classes import VectorDBCollectionsLister, ChromaConnector
from utilities.paths import CHROMA_PATH
from utilities.regex_patterns import check_input_validation
import re

# Initialize vector DBs (creates default if not present)
initialize_vector_dbs()

# Streamlit page config
st.set_page_config(page_title="Vector DB Catalog", page_icon="💾", layout="centered")

# Page title
st.title("🎓 Vector DB Catalog")

# List available collections
lister = VectorDBCollectionsLister([CHROMA_PATH])
collections_dict = lister.list_collections()

# Debug/inspection (optional)
st.subheader("📂 Available Databases & Collections")
st.json(collections_dict)

if collections_dict:
    # Layout with two side-by-side dropdowns
    db_col, collection_col = st.columns(2)

    with db_col:
        db_names = list(collections_dict.keys())
        selected_db = st.selectbox("Select a database:",options=db_names,index=None,placeholder="Choose a database...")

    with collection_col:
        if selected_db:
            collection_names = ["Create a new collection"] + collections_dict.get(selected_db, [])
             
            if collection_names:
                selected_collection = st.selectbox("Select a collection:",options=collection_names,index=None,placeholder="Choose a collection...")
            else:
                st.info("No collections found in this database.")
                selected_collection = None
        else:
            selected_collection = st.selectbox("Select a collection:",options=[],index=None,placeholder="Select a database first...",disabled=True)

    if selected_collection == "Create a new collection":
        new_collection_name_col,new_collection_metadata_col = st.columns(2)

        with new_collection_name_col:
            new_collection_name = st.text_input(label="New collection name")
        with new_collection_metadata_col:
            new_collection_description = st.text_input(label="New collection description", placeholder = "description up to 100 chars only")
            new_collection_description = str(new_collection_description)[:100]



        name_pattern_check, name_pattern_message = check_input_validation(text = new_collection_name, mode = "chars_and_numbers", allow_spaces = False)
        description_pattern_check, description_pattern_message = check_input_validation(text = new_collection_description, mode = "chars_and_numbers", allow_spaces = True)
        if not name_pattern_check:
            st.error(name_pattern_message)
        elif not description_pattern_check:
            st.error(description_pattern_message)
        elif new_collection_name in collection_names:
            st.error("Please select a valid new name")

        else:
            if st.button("create collection"):    
                chroma_db = ChromaConnector(
                            collection_name = new_collection_name,
                            db_dir = CHROMA_PATH,
                            db_type="chroma",
                            metadata = {"description":new_collection_description}
                        )


else:
    st.warning("⚠️ No databases or collections found. Please initialize or upload one.")
