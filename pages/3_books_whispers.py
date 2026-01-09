import streamlit as st
from functions.GUI import (
    upload_files_element,
    provide_google_secrets_file
)
from functions.pipeline_run import (
    parse_documents_into_md,
    split_parsed_documents_into_chunks,
    generate_chunks_embedding
    )

from functions.support_classes import GoogleDriveManager


from utilities.paths import *



# Page configuration (optional but nice)
st.set_page_config(page_title="Books Whispers", page_icon="🎓", layout="centered")

# Page title
st.title("🎓 Books Whispers")


document_parsing_flag = False
button_disabled_flag = True
_, _ , document_parsing_flag = upload_files_element(element_text = "Upload PDF documents")





if document_parsing_flag:
    google_secret_file_path, account_type = provide_google_secrets_file(account_type = "service_account")

    if google_secret_file_path and account_type:
        google_drive = GoogleDriveManager(auth_method='service_account',service_account_file=google_secret_file_path)

    button_disabled_flag = (not document_parsing_flag) and (not google_drive)

if st.button("Parse Documents", type="secondary", disabled=button_disabled_flag):
    try:
        # Create a status box
        with st.status("Starting document processing...", expanded=True) as status:

            # Step 1: Parse documents
            status.update(label="Parsing documents into markdown files...")
            parse_documents_into_md()

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")


