import streamlit as st
import os
from functions.GUI import upload_files_element
from functions.pipeline_run import parse_documents_into_md,split_parsed_documents_into_chunks
from functions.support_functions import initialize_temp_folder



_, _, _, _ = initialize_temp_folder(safe_mode = True)




# Page configuration (optional but nice)
st.set_page_config(page_title="Student Page", page_icon="🎓", layout="centered")

# Page title
st.title("🎓 Student")

document_parsing_flag = False
valid_input_docs, invalid_input_docs, document_parsing_flag = upload_files_element(element_text = "Upload PDF documents")





if st.button("Study Documents", type = "secondary", disabled= not document_parsing_flag):
    with st.status("Downloading data...", expanded=True) as status:

        # Parse documents from PDF using Smart OCR to MD files 
        st.write("Parsing documents into markdown files...")
        parse_documents_into_md()

        
        # Split the MD files into chapter or chunks (1000 tokens) with Named Entity Recognistion (NER)
        st.write("Chunking data into small chunks with metadata creation...")
        split_parsed_documents_into_chunks()
        
        
        status.update(label="Download complete!", state="complete", expanded=False)

