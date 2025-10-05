from pathlib import Path
from functions.support_classes import JSONL_Master
from functions.support_functions import list_documents
from settings.extraction_settings import SUPPORT_ONLY_PDF_FILES, image_suffixes, pdf_suffixes
from pipeline.extraction import parse_doc



def parse_documents_per_path(source_data_path, parsed_data_path ):

    supported_suffixes = pdf_suffixes if SUPPORT_ONLY_PDF_FILES else pdf_suffixes + image_suffixes
    documents_list = list_documents(folder_path = source_data_path, extensions = supported_suffixes)

    json_handler = JSONL_Master()
    parsed_dcouments = json_handler.load(file_path = r"D:\Projects\ai_agent\DataScienceRAG\Asset\parsed_documents.jsonl")
    parsed_dcouments_list = [doc['path'] for doc in parsed_dcouments] if parsed_dcouments != [] else []

    for document in documents_list:
        doc_path = document['path']

        if doc_path in parsed_dcouments_list:
            print(f"Skipping {document['name']}, since it is already parsed")
            continue

        try:
            parse_doc(doc_path, parsed_data_path, backend="pipeline")
            json_handler.save(file_path = r"D:\Projects\ai_agent\DataScienceRAG\Asset\parsed_documents.jsonl", new_data = [document])

        except Exception as e:
            raise ValueError(
                f"The file {document['name']} \n"
                f"in {doc_path}\n"
                f"could not be parsed due to {e}"
            )


