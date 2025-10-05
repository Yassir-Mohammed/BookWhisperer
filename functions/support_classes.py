import json
import os

class JSONL_Master:
    def __init__(self):
        """
        Initializes the JSONL_Master instance without loading or creating any files.
        The user must explicitly call load() or save().
        """
        self.data = []

    def load(self, file_path):
        """
        Load a JSON Lines (.jsonl) file into a list of dictionaries.
        If the file does not exist, create it as an empty file.

        Args:
            file_path (str): Path to the JSONL file.

        Returns:
            list of dict: Loaded documents.
        """
        self.data = []
        if not os.path.exists(file_path):
            # Create empty file
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            print(f"File '{file_path}' did not exist and was created as an empty JSONL file.")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:  # skip empty lines
                        self.data.append(json.loads(line))
        return self.data

    def save(self, file_path, new_data=None):
        """
        Append new documents to a JSONL file, avoiding duplicates based on 'path'.

        Args:
            file_path (str): Path to the JSONL file.
            new_data (list of dict, optional): List of document dictionaries with 'name' and 'path'.
                                                Defaults to self.data if None.
        """
        if new_data is None:
            new_data = self.data

        if not isinstance(new_data, list):
            raise TypeError("new_data must be a list of dictionaries with 'name' and 'path'.")

        # Load existing paths to avoid duplicates
        existing_paths = set(doc["path"] for doc in self.load(file_path))

        with open(file_path, "a", encoding="utf-8") as f:
            for doc in new_data:
                if doc["path"] not in existing_paths:
                    f.write(json.dumps(doc) + "\n")
                    existing_paths.add(doc["path"])
                    self.data.append(doc)
