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

    # Invalid files
    tree_lines += _format_tree_section(
        f"❌ Invalid: {invalid_count}",
        invalid_files,
        prefix_main="├──",
        branch_mid="│   ├──",
        branch_last="│   └──"
    )

    # Already committed files
    tree_lines += _format_tree_section(
        f"⚠️ Already committed: {committed_count}",
        already_committed,
        prefix_main="├──",
        branch_mid="│   ├──",
        branch_last="│   └──"
    )

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
