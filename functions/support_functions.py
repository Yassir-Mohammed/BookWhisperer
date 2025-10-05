import json
import os
from pathlib import Path

def list_documents(folder_path, extensions=['pdf']):
    """
    Scan a folder and return a list of documents with specified extensions.
    
    Args:
        folder_path (str or Path): Path to the folder to scan.
        extensions (list of str, optional): List of file extensions to include (e.g., ['.pdf', '.jpg']).
                                            Defaults to ['.pdf'] if None.
    
    Returns:
        list of dict: Each dict contains:
            - index: integer numbering the document
            - name: name of the file
            - path: full absolute path to the file
    """
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")
    
    if not extensions:
        raise ValueError("Please provide document extensions to search for.")

    extensions = [f".{ext}".lower() for ext in extensions]


    doc_list = []
    # Use sorted for consistent ordering
    for doc_path in sorted(folder_path.iterdir()):
        if doc_path.is_file() and doc_path.suffix.lower() in extensions:
            doc_list.append({
                "name": doc_path.name,
                "path": str(doc_path.resolve())
            })

    return doc_list
