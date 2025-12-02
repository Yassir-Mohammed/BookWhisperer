import re
import streamlit as st
import tempfile
import os
from functions.support_classes import JSONL_Master,ChromaConnector,VectorDBCollectionsEditor
from functions.support_functions import list_documents,build_file_summary_tree
from settings.extraction_settings import SUPPORTED_SUFFIXES
 
from utilities.paths import MAIN_DIR, TEMP_DIR, CHROMA_PATH
from utilities.regex_patterns import check_input_validation
from utilities.vectordb_initialization import initialize_vector_dbs

import inspect


def upload_files_element(element_text="Upload PDF documents",allowed_types=["pdf"],accept_multiple_files=True):
    """
    Upload multiple PDF files via Streamlit, validate filenames,
    and categorize them into:
      1. Invalid (bad names)
      2. Already committed
      3. Ready to commit (new)
    Returns (saved_files, invalid_files, committed_flag)
    """
    func_name = inspect.currentframe().f_code.co_name

    input_documents = st.file_uploader(element_text,accept_multiple_files=accept_multiple_files,type=allowed_types)
    commit_btn_disable = True

    if not input_documents:
        return {}, [], False

    # Ensure unique files only with no duplicates
    
    invalid_files = []
    valid_files = {}
    seen_filenames = set()

    for file in input_documents:
        filename = file.name

        # skip duplicate filenames
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)

        # check filename pattern
        pattern_check, _ = check_input_validation(text = filename, mode = "chars_and_numbers")
        if not pattern_check:
            invalid_files.append(filename)
        else:
            valid_files[filename] = file

    # rebuild input_documents to contain only valid, unique files
    input_documents = list(valid_files.values()) + invalid_files
    commit_btn_disable = False if valid_files != {}  else True


    if "expander_open" not in st.session_state:
        st.session_state.expander_open = True

    with st.expander("🗂 Files Commit", expanded=st.session_state.expander_open):
        msg_container = st.container()
        btn_container = st.container()

        # Load existing committed files 
        asset_json_file_tracker = os.path.join(MAIN_DIR, "Asset", "1_raw_documents.jsonl")
        json_handler = JSONL_Master()
        existing_names = set()

        if os.path.exists(asset_json_file_tracker):
            try:
                existing_records = json_handler.load(asset_json_file_tracker)
                existing_names = {doc["name"] for doc in existing_records if isinstance(doc, dict) and "name" in doc}

            except Exception as e:
                with msg_container:
                    st.warning(f"⚠️ Could not read existing JSON tracker: {e}")

        # Categorize files into: Invalid (wrongly named), Already Commited (from asset_json_file_tracker), to be commited files
        already_committed = [fn for fn in valid_files.keys() if fn in existing_names]
        new_files_to_commit = {fn: f for fn, f in valid_files.items() if fn not in existing_names}

        
        # Always show commit button at bottom
        with btn_container:
            # Get Tree-like summary of each file category
            summary_text = build_file_summary_tree(
                input_documents=input_documents,
                invalid_files=invalid_files,
                already_committed=already_committed,
                new_files_to_commit=new_files_to_commit,
            )

            st.markdown(summary_text)
 
            with btn_container:
                commit_btn = st.button("📥 Commit the documents", use_container_width=True, disabled= commit_btn_disable)

        # Commit the temporary saved files into 1_raw_data for processing
        if commit_btn:

            RAW_DATA_DIR = os.path.join(TEMP_DIR, "1_raw_data")
            os.makedirs(RAW_DATA_DIR, exist_ok=True)

            saved_files = {}
            for filename, file in new_files_to_commit.items():
                save_path = os.path.join(RAW_DATA_DIR, filename)
                try:
                    print(f"{func_name}: committing {filename}")
                    with open(save_path, "wb") as f:
                        f.write(file.read())
                    saved_files[filename] = save_path
                except Exception as exc:
                    print(f"{func_name}: Failed to save {filename}: {exc}")


            # Update JSON tracker
            documents_dict = list_documents(folder_path=RAW_DATA_DIR, extensions=SUPPORTED_SUFFIXES)
            json_handler.save(file_path=asset_json_file_tracker, new_data=documents_dict)

            with msg_container:
                st.success(f"✅ {len(saved_files)} new PDF file(s) committed successfully to `{RAW_DATA_DIR}`")
                print(f"{func_name}: {len(saved_files)} new PDF file(s) committed successfully to `{RAW_DATA_DIR}`")

            # Collapse the expander
            st.session_state.expander_open = False

            return saved_files, invalid_files, True

    return {}, invalid_files, False



    
def create_or_select_collection(vectorDB_path:str, create_collection_automatically:bool):

    func_name = inspect.currentframe().f_code.co_name

    initialize_vector_dbs()
    was_created = False
    # List available collections
    lister = VectorDBCollectionsEditor([CHROMA_PATH])
    collections_dict = lister.list_collections()


        
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
       
            new_collection_name_col, new_collection_metadata_col = st.columns(2)

            with new_collection_name_col:
                new_collection_name = st.text_input(label="New collection name")
            with new_collection_metadata_col:
                new_collection_description = st.text_input(label="New collection description", placeholder = "description up to 100 chars only")
                new_collection_description = str(new_collection_description)[:100]



            name_pattern_check, _ = check_input_validation(text = new_collection_name, mode = "chars_and_numbers", allow_spaces = False, 
                                                           print_errors = True, error_suffix = "Collection Name")
            
            description_pattern_check, _ = check_input_validation(text = new_collection_description, mode = "chars_and_numbers", allow_spaces = True,
                                                                   print_errors = True, error_suffix = 'Description')
            
            # Validate inputs
            button_disabled = (name_pattern_check) and (description_pattern_check) and not(new_collection_name in collection_names)
            
            if new_collection_name in collection_names: 
                st.error("Please select a valid new name")
            
            
            if (create_collection_automatically) and (button_disabled): 
                create_new_collection(new_collection_name = new_collection_name, vectorDB_path = vectorDB_path, metadata = {"description": new_collection_description})
                was_created = True
                        
            else:
                if st.button("create collection", disabled = not button_disabled):    
                    create_new_collection(new_collection_name = new_collection_name, vectorDB_path = vectorDB_path, metadata = {"description": new_collection_description})
                    was_created = True



    else:
        st.warning("⚠️ No databases or collections found. Please initialize or upload one.")
     
        
    return new_collection_name if was_created else (selected_collection if selected_collection != "New collection name" else None)


def create_new_collection(*,new_collection_name:str,vectorDB_path:str, metadata:dict):
    
    func_name = inspect.currentframe().f_code.co_name
    succeeded = False
    try:
        chroma_db = ChromaConnector(
            collection_name=new_collection_name,
            db_dir=vectorDB_path,
            db_type="chroma",
            metadata=metadata
        )
        succeeded = True
    except Exception as exc:
        print(f"{func_name}: Could not create Chroma connector: {exc}")
        succeeded = False
    return succeeded


def collection_creation_expander(vectorDB_path:str, create_collection_automatically:bool = True):
    with st.expander("Select or Create Collection"):
        return create_or_select_collection(vectorDB_path = vectorDB_path, create_collection_automatically = create_collection_automatically)