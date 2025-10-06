from functions.support_classes import JSONL_Master, MarkdownBookProcessor
from functions.support_functions import list_documents,list_md_documents
from settings.extraction_settings import SUPPORT_ONLY_PDF_FILES, image_suffixes, pdf_suffixes
from pipeline.extraction import parse_doc
from pathlib import Path


def parse_documents_per_path(source_data_path, parsed_data_path ):


    asset_json_file_tracker = r"D:\Projects\ai_agent\DataScienceRAG\Asset\parsed_documents.jsonl"

    supported_suffixes = pdf_suffixes if SUPPORT_ONLY_PDF_FILES else pdf_suffixes + image_suffixes
    
    documents_list = list_documents(folder_path = source_data_path, extensions = supported_suffixes)
    print("Reading Documents...")

    json_handler = JSONL_Master()
    parsed_dcouments = json_handler.load(file_path = asset_json_file_tracker)
    parsed_dcouments_list = [Path(doc['path']) for doc in parsed_dcouments] if parsed_dcouments != [] else []

    for document in documents_list:
        doc_name = document['name']
        doc_path = Path(document['path'])

        if doc_path in parsed_dcouments_list:
            print(f"Skipping {document['name']}, since it is already parsed")
            continue
        else:
            print(f"Parsing {doc_name}")

        try:
            
            parse_doc([doc_path], parsed_data_path, backend="pipeline")
            json_handler.save(file_path = asset_json_file_tracker, new_data = [document])

        except Exception as e:
            raise ValueError(
                f"The file {doc_name} \n"
                f"in {doc_path}\n"
                f"could not be parsed due to {e}"
            )

    print("Parsing documents is finished.")

def split_parsed_documents_into_chunks(source_data_path,split_data_path):

    asset_json_file_tracker = r"D:\Projects\ai_agent\DataScienceRAG\Asset\cleaned_documents.jsonl"

    md_documents = list_md_documents(folder_path= source_data_path)
    print("Finding md documents...")

    json_handler = JSONL_Master()
    split_documents = json_handler.load(file_path = asset_json_file_tracker)
    split_dcouments_list = [Path(doc['path']) for doc in split_documents] if split_documents != [] else []

    for document in md_documents:
       
        try:
            doc_name = document.get('name',None)
            doc_path = document.get('path', None)
            
            if doc_path in split_dcouments_list:
                print(f"Skipping {doc_name}, since it is already splitted")
                continue
            else:
                print(f"Splitting {doc_name}")

            processor = MarkdownBookProcessor(doc_path)
            processor.split_into_chapters()                     # Split chapters or fallback chunks
            processor.extract_entities()                        # Detect characters & locations
            processor.save_chapters_as_json(str(split_data_path))            # Save cleaned chapters

            json_handler.save(file_path = asset_json_file_tracker, new_data = [document])

        except Exception as e:
            raise ValueError(
                f"The file {doc_name} \n"
                f"in {doc_path}\n"
                f"could not be splitted due to {e}"
            )
    print("Splitting documents is finished.")