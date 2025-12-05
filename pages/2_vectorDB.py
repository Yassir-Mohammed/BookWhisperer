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


    vectorDB = VectorDBCollectionsEditor(db_paths = [CHROMA_PATH])
    all_collections = vectorDB.list_collections()


    to_be_deleted_collections = st.multiselect("Select Collection to Delete",options = all_collections.get("chroma"))

    if st.button("Delete Collection"):
            for collection in to_be_deleted_collections:
                try:
                    chroma_db = ChromaConnector(collection_name=collection,db_dir=CHROMA_PATH,db_type="chroma")
                        
                    success = chroma_db.delete_collection(collection_name=collection)
                    if success:
                        st.success(f"Collection '{collection}' was deleted.")
                    else:
                        st.error(f"Could not delete collection '{collection}'.")
                except Exception as exc:
                    st.error(f"Error while deleting collection: {exc}")
        
   

        

