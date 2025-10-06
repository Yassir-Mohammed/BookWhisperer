

from functions.pipeline_run import parse_documents_per_path,split_parsed_documents_into_chunks



raw_data_path = r"D:\Projects\ai_agent\Data\raw_data"
parsed_data_path = r"D:\Projects\ai_agent\Data\parsed_data"
split_data_path = r"D:\Projects\ai_agent\Data\clean_data"


parse_documents_per_path(source_data_path = raw_data_path, parsed_data_path = parsed_data_path)

split_parsed_documents_into_chunks(source_data_path = parsed_data_path, split_data_path = split_data_path)

