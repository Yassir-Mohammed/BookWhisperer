from functions.support_classes import JSONL_Master, MarkdownBookProcessor, ChapterLoader, ChromaConnector
from functions.support_functions import list_documents,list_md_documents, chunk_text, estimate_batch_size, get_key_value, build_id
from settings.extraction_settings import SUPPORTED_SUFFIXES
from settings.transformation_settings import SEQUENCE_LENGTH, TEXT_OVERLAP_RATIO, EMBEDDING_MODELS,NORMALIZE_EMBEDDINGS
from utilities.paths import *
from pipeline.extraction import parse_doc
from pathlib import Path
import os
import streamlit as st
import inspect

def parse_documents_into_md():
    
    func_name = inspect.currentframe().f_code.co_name
    print(f"{func_name}: Reading Documents...")
    documents_list = list_documents(folder_path = RAW_DATA_DIR, extensions = SUPPORTED_SUFFIXES)
    
    asset_json_file_tracker = os.path.join(MAIN_DIR,"Asset","2_parsed_documents.jsonl")
    json_handler = JSONL_Master()
    parsed_dcouments = json_handler.load(file_path = asset_json_file_tracker)
    parsed_dcouments_list = [Path(doc['path']) for doc in parsed_dcouments] if parsed_dcouments != [] else []

    total_documents = len(documents_list) - 1
    for i, document in enumerate(documents_list):

        
        doc_name = document['name']
        doc_path = Path(document['path'])

        if doc_path in parsed_dcouments_list:
            print(f"{func_name}: Skipping {document['name']}, since it is already parsed")
            continue
        else:
            print(f"{func_name}: Parsing {doc_name} ({i}/{total_documents})")

        try:
            
            parse_doc(path_list = [doc_path], output_dir = PARSED_DATA_DIR, backend="pipeline")
            json_handler.save(file_path = asset_json_file_tracker, new_data = [document])

        except Exception as exc:
            raise ValueError(
                f"{func_name}:\n"
                f"The file {doc_name} \n"
                f"in {doc_path}\n"
                f"could not be parsed due to {exc}"
            )

    print(f"{func_name}: Parsing documents is finished.")


def split_parsed_documents_into_chunks():

    func_name = inspect.currentframe().f_code.co_name

    asset_json_file_tracker = os.path.join(MAIN_DIR,"Asset","3_cleaned_documents.jsonl")
       
    print(f"{func_name}: Finding md documents...")
    md_documents = list_md_documents(folder_path = PARSED_DATA_DIR)
    

    json_handler = JSONL_Master()
    split_documents = json_handler.load(file_path = asset_json_file_tracker)
    split_dcouments_list = [Path(doc['path']) for doc in split_documents] if split_documents != [] else []
    
    total_documents = len(md_documents) - 1
    for i, document in enumerate(md_documents):
       
        try:
            doc_name = document.get('name',None)
            doc_path = document.get('path', None)
            
            if doc_path in split_dcouments_list:
                print(f"{func_name}: Skipping {doc_name}, since it is already splitted")
                continue
            else:
                print(f"{func_name}: Splitting {doc_name} ({i}/{total_documents})")

            processor = MarkdownBookProcessor(doc_path)
            processor.split_into_chapters()                     # Split chapters or fallback chunks
            processor.extract_entities()                        # Detect characters & locations
            processor.save_chapters_as_json(str(SPLIT_DATA_DIR))            # Save cleaned chapters

            json_handler.save(file_path = asset_json_file_tracker, new_data = [document])

        except Exception as exc:
            raise ValueError(
                f"{func_name}:\n"
                f"The file {doc_name} \n"
                f"in {doc_path}\n"
                f"could not be splitted due to {exc}"
            )
    print(f"{func_name}: Splitting documents is finished.")



def generate_chunks_embedding(*,collection_name,  model_name = EMBEDDING_MODELS[0], vector_db = 'chroma'):

    func_name = inspect.currentframe().f_code.co_name

    if vector_db == 'chroma':
        chroma_db = ChromaConnector(collection_name = collection_name, db_dir = CHROMA_PATH, db_type = vector_db)
    else:
        raise ValueError(f"{func_name}:Unsupported vector_db value: {vector_db}. Expected 'chroma'.")

    
    from sentence_transformers import SentenceTransformer

    # Embedding Model loading
    model = SentenceTransformer(model_name.get("model_name"))
    max_batch_size, device = estimate_batch_size(model)


    asset_json_file_tracker = os.path.join(MAIN_DIR,"Asset","4_vectorized_documents.jsonl")

    json_handler = JSONL_Master()
    json_documents = json_handler.load(file_path = asset_json_file_tracker)
    json_documents_list = [doc['path'] for doc in json_documents] if json_documents != [] else []
    


    # Chunks explorer 
    loader = ChapterLoader(SPLIT_DATA_DIR)
    files = loader.list_files()
    total_documents = len(files) - 1
    
    for j,json_file in enumerate(files):
        

        json_name , file_path = json_file.get("name"), json_file.get("path")

        if json_file in json_documents_list:
            print(f"{func_name}: Skipping {json_name}, since it is already vectorized")
            continue
    

        data = loader.load_json(file_path)
            
        book_title = get_key_value(dictionary = data, key="book_title", expected_type=str)
        text = get_key_value(dictionary = data, key="body", expected_type=str)

        section_title = data.get("title") or ""
        start_index = int(data.get("start") or 0)
        end_index = int(data.get("end") or 0)
        word_count = int(data.get("word_count") or 0)
        
        entities = data.get("entities") or {} 
        flat_entities = {k: ", ".join(v) if v else "" for k, v in entities.items()} if entities else {}



        print(f"{func_name}: Chunking file ({j}/{total_documents})")
               
        # Split chunks (chapters) into SEQUENCE_LENGTH-sized text with overlap of TEXT_OVERLAP_RATIO%
        chunks = chunk_text(text, seq_len = SEQUENCE_LENGTH, overlap_ratio = TEXT_OVERLAP_RATIO)
        
        total_chunks = len(chunks) - 1
        for i, chunk_text_str in enumerate(chunks):
            
            
            unique_id = build_id(book_title,section_title,i)

            meta_data = {
                "start_index": start_index,
                "end_index": end_index,
                "word_count": word_count,
                **flat_entities
            }

            
            try:
                print(f"{func_name}: generating embedding of chunk ({i}/{total_chunks})")
                embeddings = model.encode(
                    chunk_text_str,
                    normalize_embeddings=NORMALIZE_EMBEDDINGS,
                    show_progress_bar=False,
                    batch_size=max_batch_size,
                    device=device
                )
            except Exception as exc:
                print(f"{func_name}: failed while generating embedding for chunk ({i}/{total_chunks})\n")
                print(f"Error: {exc}")
                raise
            
            try:
                print(f"{func_name}: writing chunk ({i}/{total_chunks}) embedding to vectorDB\n")
                chroma_db.add_documents(
                    ids=[unique_id],
                    embeddings=[embeddings],
                    metadatas=[meta_data]
                )
                json_handler.save(file_path = asset_json_file_tracker, new_data = [json_file])
            except Exception as exc:
                print(f"{func_name}: failed to write chunk ({i}/{total_chunks})\n")
                print(f"ID: {unique_id}")
                print(f"Error: {exc}")
        


    return 


