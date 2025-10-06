

from functions.pipeline_run import parse_documents_per_path,split_parsed_documents_into_chunks



if __name__ == "__main__":

    raw_data_path = r"D:\Projects\ai_agent\Data\raw_data"
    parsed_data_path = r"D:\Projects\ai_agent\Data\parsed_data"
    split_data_path = r"D:\Projects\ai_agent\Data\clean_data"

    # Parse documents from PDF using Smart OCR to MD files 
    parse_documents_per_path(source_data_path = raw_data_path, parsed_data_path = parsed_data_path)

    # Split the MD files into chapter or chunks (1000 tokens) with Named Entity Recognistion (NER)
    split_parsed_documents_into_chunks(source_data_path = parsed_data_path, split_data_path = split_data_path)

