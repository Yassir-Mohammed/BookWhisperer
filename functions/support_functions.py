from pathlib import Path
import os
import re

from settings.transformation_settings import SEQUENCE_LENGTH, TEXT_OVERLAP_RATIO


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
                if file_path.suffix.lower() in extensions:
                    doc_list.append({
                        "name": file_path.name,
                        "path": Path(str(file_path.resolve()))
                    })
                    

    return doc_list



def build_file_summary_tree(input_documents, invalid_files, already_committed, new_files_to_commit):
    """
    Build a hierarchical tree summary of uploaded files and their categories.
    Returns a formatted string (Markdown-safe) that can be passed to st.info().
    """

    def _format_tree_section(title, items, prefix_main, branch_mid, branch_last):
        lines = [f"{prefix_main} {title}"]
        if not items:
            lines.append(f"{branch_last} (none)")
            return lines
        for i, name in enumerate(items):
            if len(items) == 1:
                lines.append(f"{branch_last} {name}")
            else:
                child_prefix = branch_mid if i < len(items) - 1 else branch_last
                lines.append(f"{child_prefix} {name}")
        return lines

    total_uploaded = len(input_documents)
    invalid_count = len(invalid_files)
    committed_count = len(already_committed)
    ready_count = len(new_files_to_commit)

    # Start the tree
    tree_lines = [f"📦 Uploaded files: {total_uploaded}"]

    if invalid_count > 0:
        # Invalid files
        tree_lines += _format_tree_section(
            f"❌ Invalid: {invalid_count}",
            invalid_files,
            prefix_main="├──",
            branch_mid="│   ├──",
            branch_last="│   └──"
        )
    if committed_count > 0:
        # Already committed files
        tree_lines += _format_tree_section(
            f"⚠️ Already committed: {committed_count}",
            already_committed,
            prefix_main="├──",
            branch_mid="│   ├──",
            branch_last="│   └──"
        )

    if ready_count > 0:
        # Ready to commit files (final branch)
        lines_ready = _format_tree_section(
            f"🆕 Ready to commit: {ready_count}",
            list(new_files_to_commit.keys()),
            prefix_main="└──",
            branch_mid="    ├──",
            branch_last="    └──"
        )
        tree_lines += lines_ready

    tree_text = "\n".join(tree_lines)

    # Return clean, formatted string
    summary_text = (
        
        f"```\n{tree_text}\n```\n"
        f"Allowed characters: letters, numbers, underscores (_), hyphens (-), or dots (.) only."
    )

    return summary_text


def chunk_text(text, seq_len=SEQUENCE_LENGTH, overlap_ratio=TEXT_OVERLAP_RATIO):
    """
    Split text into overlapping chunks.
    Args:
        text (str): input text.
        seq_len (int): max tokens/words per chunk.
        overlap_ratio (float): fraction of overlap between chunks.
    Returns:
        List[str]: list of text chunks.
    """
    words = text.split()
    if not words:
        return []
    
    overlap = int(seq_len * overlap_ratio)
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + seq_len, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += seq_len - overlap

    return chunks

def estimate_batch_size(model, sample_text="test text"):
    
    import psutil
    from sentence_transformers import SentenceTransformer

    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    # Rough size per embedding vector
    dummy = model.encode([sample_text])
    vector_size = dummy[0].nbytes

    #batch about 5 percent of free memory
    memory_fraction = 0.05

    if gpu_available:
        free_bytes = torch.cuda.mem_get_info()[0]
    else:
        free_bytes = psutil.virtual_memory().available

    budget = free_bytes * memory_fraction
    batch = int(budget // vector_size)

    if batch < 1:
        batch = 1
    if batch > 512:
        batch = 512

    return batch,"cuda" if gpu_available else "cpu"

def get_key_value(dictionary:dict, key:str|int, expected_type=str):
    value = dictionary.get(key)

    if value is None or value == "":
        raise ValueError(f"Missing required field: {key}")

    # Validate or convert type
    try:
        if expected_type is str:
            return str(value)
        if expected_type is int:
            return int(value)
        if expected_type is float:
            return float(value)
    except Exception:
        raise ValueError(f"Field {key} must be of type {expected_type.__name__}")

    # Fallback if type not recognized
    return value


def build_id(*parts):
    cleaned = [str(p) for p in parts if p not in (None, "")]
    return "_".join(cleaned)
