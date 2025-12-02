import streamlit as st
from functions.GUI import upload_files_element,create_or_select_collection
from utilities.vectordb_initialization import initialize_vector_dbs
from functions.support_classes import VectorDBCollectionsEditor, ChromaConnector
from utilities.paths import CHROMA_PATH




selected_collection = None

# Streamlit page config
st.set_page_config(page_title="Vector DB Catalog", page_icon="💾", layout="centered")

# Page title
st.title("🎓 Vector DB Catalog")

selected_collection = create_or_select_collection(vectorDB_path = CHROMA_PATH, create_collection_automatically = False)



if (selected_collection) and (selected_collection != "Create a new collection"):
    st.subheader("📂 Edit Databases & Collections")

    chroma_db = ChromaConnector(collection_name = selected_collection, db_type="chroma", db_dir = CHROMA_PATH)
        
    collection_data = chroma_db.get_collection_data()
        

    to_be_deleted = st.multiselect(label = "To be deleted records", options  = collection_data)
        

