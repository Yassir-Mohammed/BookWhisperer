from pathlib import Path
import os
import re


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



def list_md_documents(folder_path, extensions=['md']):
    """
    Scan a folder and return a list of documents with specified extensions
    where the folder or file name contains uppercase letters (pattern in UPPERCASE).

    Args:
        folder_path (str or Path): Path to the folder to scan.
        extensions (list of str, optional): File extensions to include (e.g., ['md']).
                                            Defaults to ['md'].

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
    
    extensions = [f".{ext.lower()}" if not ext.startswith('.') else ext.lower() for ext in extensions]

    doc_list = []
    

    for root, _, files in os.walk(folder_path):
        root_path = Path(root)
        # Only consider paths/folders with uppercase pattern
        if any(re.search(r'[A-Z]', part) for part in root_path.parts):
            for file in sorted(files):
                file_path = root_path / file
                if file_path.suffix.lower() in extensions and re.search(r'[A-Z]', file_path.stem):
                    doc_list.append({
                        "name": file_path.name,
                        "path": Path(str(file_path.resolve()))
                    })
                    

    return doc_list
