import streamlit as st
from functions.GUI import upload_files_element
from functions.pipeline_run import (
    parse_documents_into_md,
    split_parsed_documents_into_chunks,
    generate_chunks_embedding
    )


from utilities.paths import *






# Page configuration (optional but nice)
st.set_page_config(page_title="Student Page", page_icon="🎓", layout="centered")

# Page title
st.title("🎓 Student")

st.markdown(CHROMA_PATH)
document_parsing_flag = False
_, _ , document_parsing_flag = upload_files_element(element_text = "Upload PDF documents")


if st.button("Study Documents", type="secondary", disabled=not document_parsing_flag):
    try:
        # Create a status box
        with st.status("Starting document processing...", expanded=True) as status:

            # Step 1: Parse documents
            status.update(label="Parsing documents into markdown files...")
            parse_documents_into_md()

            # Step 2: Chunk documents
            status.update(label="Chunking data into small chunks with metadata creation...")
            split_parsed_documents_into_chunks()


            # step 3: read the chunks, embedd them, and store them into VectroDB
            status.update(label="Embedding then storing document(s)...")
            generate_chunks_embedding(collection_name = 'test_Collection', vector_db="chroma")

            # Final update
            status.update(label="✅ Processing complete!", state="complete", expanded=False)

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")


