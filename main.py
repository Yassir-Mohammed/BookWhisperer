from pathlib import Path
from settings.extraction_settings import SUPPORT_ONLY_PDF_FILES, image_suffixes, pdf_suffixes
from functions.pipeline_run import parse_documents_per_path



raw_data_path = r"D:\Projects\ai_agent\Data\raw_data"
parsed_data_path = r"D:\Projects\ai_agent\Data\parsed_data"



parse_documents_per_path(source_data_path = raw_data_path, parsed_data_path = parsed_data_path)

