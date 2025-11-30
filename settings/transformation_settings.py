from typing import Dict


NER_DEFAULT_TYPES:list[str] = ["PERSON", "FAC", "ORG", "GPE", "LOC", "PRODUCT","EVENT", "QUANTITY", "ORDINAL", "CARDINAL"]
NER_DEFAULT_TOP_N: int | None = None
NER_DEFAULT_EXTENSION_TO_REMOVE:set[str] = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg",".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",".txt", ".csv", ".zip", ".rar", ".html", ".xml", ".json"}
CHUNK_SIZE:int = 1000
TEXT_OVERLAP_RATIO:float = 0.3
SEQUENCE_LENGTH:int = 500
EMBEDDING_MODELS:list[Dict[str, str]] = [
                                        {'model_name':"BAAI/bge-base-en", 'description':""},
                                        {'model_name':"BAAI/bge-m3", 'description':""}
                                        ]

NORMALIZE_EMBEDDINGS: bool = True
