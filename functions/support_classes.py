# Initialization + default paths
from utilities.paths import *

# Settings 
from settings.transformation_settings import *
from settings.vector_db_settings import DEFAULT_DB_TYPES

# General Imports
import json
import os
import re
from typing import List, Dict, Any
from collections import Counter
from pathlib import Path
from abc import ABC, abstractmethod
import warnings

# NLP Import 
import spacy

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions




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

        
        if not isinstance(new_data, list):
            raise TypeError("new_data must be a list of dictionaries with 'name' and 'path' keys.")

        # Check that each element in the list is a dictionary with required keys
        for i, item in enumerate(new_data):
            if not isinstance(item, dict):
                raise TypeError(f"Item at index {i} is not a dictionary: {type(item).__name__}")
            
            required_keys = {"name", "path"}
            missing_keys = required_keys - set(item.keys())
            
            if missing_keys:
                raise ValueError(f"Item at index {i} is missing required keys: {', '.join(missing_keys)}")
    

        # Load existing paths to avoid duplicates
        existing_paths = set(doc["path"] for doc in self.load(file_path))

        with open(file_path, "a", encoding="utf-8") as f:
           
            for doc in new_data:
                if doc["path"] not in existing_paths:
                    f.write(json.dumps(doc, default=str) + "\n")
                    existing_paths.add(str(doc["path"]))
                    self.data.append(doc)


class MarkdownBookProcessor:
    _shared_nlp = None  # class-level shared spaCy model

    def __init__(self, md_path: str, keep_headers_in_body: bool = False, nlp=None):
        """
        Initialize with either a provided spaCy model or a shared one (loaded once).
        """
        self.md_path = str(md_path)
        self.keep_headers_in_body = keep_headers_in_body

        # Use shared or provided NLP model
        if nlp:
            self.nlp = nlp
        else:
            self.nlp = self._get_shared_nlp()

        self.text = self._load_file()
        self.chapters = []

    @classmethod
    def _get_shared_nlp(cls):
        """Load the shared spaCy model once and reuse it across all instances."""
        if cls._shared_nlp is None:
            import spacy
            print("Loading spaCy model (once)...")
            cls._shared_nlp = spacy.load("en_core_web_sm")
        return cls._shared_nlp
    # -------------------------
    # Internal helper methods
    # -------------------------
    def _load_file(self) -> str:
        with open(self.md_path, "r", encoding="utf-8-sig") as f:
            return f.read()

    def _clean_header_text(self, s: str) -> str:
        return re.sub(r'[\s:]+$', '', s).strip()


    
    def _clean_text_for_embedding(self, text: str) -> str:

        # Remove image references like (images/XXXX.jpg) where XXXX can be anything
        text = re.sub(r"\(images/[^)]+\.(?:jpg|jpeg|png|gif|bmp|tiff|svg)\)", "", text, flags=re.IGNORECASE)
        # Replace tabs with space and normalize newlines
        text = text.replace("\t", " ").replace("\n", " ")
        # Remove excessive markdown symbols
        text = re.sub(r"[*_`~#>]+", "", text)
        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text


    def _make_safe_filename(self, s: str) -> str:
        """
        Make a filename space-friendly, readable, lowercase, remove special chars except dash/space.
        """
        s = s.strip()
        s = re.sub(r'[\\/:"*?<>|]+', '', s)  # remove illegal filesystem chars
        s = re.sub(r'\s+', ' ', s)           # normalize spaces
        s = s.replace(' ', ' ')               # keep spaces for readability
        return s[:60]  # limit length

    def _build_chapters_from_splits(self, splits: List[Dict], book_title:str) -> List[Dict]:
        splits = sorted(splits, key=lambda h: h["start"])
        chapters = []

        # optional preamble
        if splits and splits[0]["start"] > 0:
            pre_text = self._clean_text_for_embedding(self.text[:splits[0]["start"]].strip())
            if len(pre_text.split()) >= 20:
                chapters.append({
                    "book_title": book_title,
                    "title": "Preamble",
                    "level": None,
                    "start": 0,
                    "end": splits[0]["start"],
                    "word_count": len(pre_text.split()),
                    "body": pre_text 
                })

        for i, sp in enumerate(splits):
            start = sp["start"] if self.keep_headers_in_body else sp["end"]
            end = splits[i + 1]["start"] if i + 1 < len(splits) else len(self.text)
            body = self._clean_text_for_embedding(self.text[start:end].strip())
            chapters.append({
                "book_title": book_title,
                "title": sp.get("text") or f"Chapter {i+1}",
                "level": sp.get("level"),
                "start": start,
                "end": end,
                "word_count": len(body.split()),
                "body": body    
            })
        return chapters

    # -------------------------
    # Public methods
    # -------------------------
    def split_into_chapters(self):
        book_title = str(os.path.splitext(os.path.basename(self.md_path))[0])

        # Find markdown headers
        header_re = re.compile(r'^(?P<hash>#{1,6})\s*(?P<text>.+?)\s*$', re.MULTILINE)
        headers = [{"start": m.start(), "end": m.end(), "level": len(m.group("hash")),
                    "text": self._clean_header_text(m.group("text"))} for m in header_re.finditer(self.text)]

        # 1) Headers with "chapter"
        chapter_headers = [h for h in headers if re.search(r'\bchapter\b', h["text"], re.IGNORECASE)]
        if len(chapter_headers) >= 2:
            self.chapters = self._build_chapters_from_splits(chapter_headers, book_title)
            # add book title
            #for ch in self.chapters:ch["book_title"] = book_title
            return self.chapters

        # 2) High-level headers fallback
        high_level_headers = [h for h in headers if h["level"] in (1, 2)]
        if len(high_level_headers) >= 2:
            self.chapters = self._build_chapters_from_splits(high_level_headers, book_title)
            #for ch in self.chapters:ch["book_title"] = book_title
            return self.chapters

        # 3) Plain-text "CHAPTER" lines fallback
        chap_line_re = re.compile(r'^(?P<header>\s*chapter\b[^\n]*)', re.IGNORECASE | re.MULTILINE)
        chap_lines = [{"start": m.start(), "end": m.end(), "level": None,
                       "text": self._clean_header_text(m.group("header"))} for m in chap_line_re.finditer(self.text)]
        if len(chap_lines) >= 2:
            self.chapters = self._build_chapters_from_splits(chap_lines, book_title)
            #for ch in self.chapters:ch["book_title"] = book_title
            return self.chapters

        # 4) Last fallback: split into fixed-size chunks
        words = self.text.split()
        chunk_size = CHUNK_SIZE
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunks.append({
                "book_title": book_title,
                "chapter": f"Chunk {(i // chunk_size) + 1}",
                "level": None,
                "start": None,
                "end": None,
                "word_count": len(chunk_words),
                "body": self._clean_text_for_embedding(" ".join(chunk_words)),
                
            })
        self.chapters = chunks
        return self.chapters


    def extract_entities(self,entity_types: list = NER_DEFAULT_TYPES,top_n: int = NER_DEFAULT_TOP_N):
        """
        Extract named entities from chapters with:
        - normalization and deduplication
        - stopword/short token cleanup
        - overlap resolution (PERSON dominates)
        - possessive/suffix stripping ("Harry's" → "harry")
        - lowercase normalization
        - removal of entities with file extensions (e.g., .jpg, .png)
        """
        # Validate allowed entity labels
        permitted_entities = set(self.nlp.get_pipe("ner").labels)
        valid_entity_types = [e for e in entity_types if e in permitted_entities]
        if not valid_entity_types:
            raise ValueError(f"No valid entity types provided. Permitted: {permitted_entities}")

        stopwords = self.nlp.Defaults.stop_words
        file_extensions = NER_DEFAULT_EXTENSION_TO_REMOVE

        def _normalize_entity(ent: str) -> str:
            """Normalize entity: lowercase, remove possessive/apostrophe-s and plural 's'."""
            ent = ent.strip().lower()
            ent = re.sub(r"'s\b", "", ent)   # remove possessive ('Harry's' → 'harry')
            ent = re.sub(r"s\b", "", ent)    # remove trailing plural s ('Potters' → 'potter')
            return ent.strip()

        def _has_file_extension(ent: str) -> bool:
            """Check if entity contains any known file extension."""
            for ext in file_extensions:
                if ent.endswith(ext) or f"{ext} " in ent or f" {ext}" in ent:
                    return True
            return False

        def _is_valid_entity(ent: str) -> bool:
            """Filter out stopwords, digits, punctuation-only, short, or file-like entities."""
            if not ent or len(ent) < 2:
                return False
            if ent in stopwords:
                return False
            if ent.isdigit():
                return False
            if _has_file_extension(ent):
                return False
            if all(ch in ".,!?;:-_–—'\"()[]{}" for ch in ent):
                return False
            return True

        for ch in self.chapters:
            doc = self.nlp(ch["body"])
            results = {}

            # Step 1: Extract and normalize entities
            for ent_type in valid_entity_types:
                ents = [
                    _normalize_entity(ent.text)
                    for ent in doc.ents
                    if ent.label_ == ent_type and _is_valid_entity(ent.text.lower())
                ]
                freq = Counter(ents)
                unique = []
                for name, _ in freq.most_common():
                    if not any(name in existing or existing in name for existing in unique):
                        unique.append(name)
                if top_n:
                    unique = unique[:top_n]
                results[ent_type] = unique

            # Step 2: Remove overlaps and suffix variants based on PERSON
            if "PERSON" in results:
                person_names = {p.lower() for p in results["PERSON"]}
                expanded_person_names = set(person_names)
                for name in list(person_names):
                    if name.endswith("s"):
                        expanded_person_names.add(name[:-1])
                    if name.endswith("'s"):
                        expanded_person_names.add(name[:-2])

                for ent_type in list(results.keys()):
                    if ent_type != "PERSON":
                        results[ent_type] = [
                            ent for ent in results[ent_type]
                            if ent not in expanded_person_names
                        ]

            # Step 3: Ensure all entities are lowercase and unique
            for ent_type in results:
                results[ent_type] = sorted(set(e.lower() for e in results[ent_type]))

            ch["entities"] = results

        return self.chapters




    def save_chapters_as_json(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.md_path))[0]
       
        for i, ch in enumerate(self.chapters, 1):
            
            safe_title = self._make_safe_filename(ch["book_title"])
            filename = f"{i:02d} - {base_name} - {safe_title}.json"
            filepath = os.path.join(output_dir, filename)

            # Save the full chapter dict as JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(ch, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.chapters)} chapters as JSON to {output_dir}")


class ChapterLoader:
    
    """
    Loads and manages JSON files from a given directory.
    Each JSON is expected to contain a dictionary (e.g., chapter data).
    """

    def __init__(self, base_path: str, recursive: bool = False):
        """
        Initialize the loader with the base directory.
        Args:
            base_path (str): Path to the directory containing JSON files.
            recursive (bool): Whether to search subdirectories as well.
        """
        self.base_path = os.path.abspath(base_path)
        self.recursive = recursive
        self.json_files: List[Dict[str, str]] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """
        Scan the directory for JSON files and store them in a list.
        Each entry: {"name": filename.json, "path": full_path}.
        """
        files_found = []
        if self.recursive:
            for root, _, files in os.walk(self.base_path):
                for f in files:
                    if f.lower().endswith(".json"):
                        files_found.append({
                            "name": f,
                            "path": os.path.join(root, f)
                        })
        else:
            for f in os.listdir(self.base_path):
                if f.lower().endswith(".json"):
                    files_found.append({
                        "name": f,
                        "path": os.path.join(self.base_path, f)
                    })
        self.json_files = files_found

    def list_files(self) -> List[Dict[str, str]]:
        """Return a list of detected JSON files: [{'name': ..., 'path': ...}, ...]."""
        return self.json_files

    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load and return the dictionary stored in the specified JSON file.
        Args:
            file_path (str): The full path to the JSON file.
        Returns:
            dict: The JSON content as a dictionary.
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the JSON content isn't a dictionary.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No file found at path '{file_path}'")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected a dictionary in '{file_path}', got {type(data)} instead.")
        return data
    


    
# Parent class
class VectorDBConnector(ABC):

    

    def __init__(self,*, db_dir:str, collection_name: str, db_type: str = "chroma", overwrite: bool = False, **kwargs):
        """
        Base constructor for vector DB connectors.
        Args:
            collection_name (str): Name of the collection.
            db_type (str): Type of vector DB ('chroma' or 'leann').
            overwrite (bool): Whether to overwrite an existing collection.
            kwargs: DB-specific parameters.
        """
        self.collection_name = collection_name
        self.db_type = db_type.lower()
        self.kwargs = kwargs
        self.connection = None
        self.collection = None
        self.db_dir = db_dir

        # Validate database type
        if self.db_type not in DEFAULT_DB_TYPES:
            raise ValueError(f"Invalid db_type '{self.db_type}'. Must be one of {DEFAULT_DB_TYPES}.")


        # Connect automatically
        self.connect()

        # Always create collection
        self.create_collection(overwrite=overwrite)

    @abstractmethod
    def connect(self):
        """Connect to the vector database."""
        pass

    @abstractmethod
    def create_collection(self, overwrite: bool = False):
        """Create or get a collection."""
        pass

    @abstractmethod
    def add_documents(self, ids, embeddings, metadatas):
        """Add pre-computed embeddings to the collection."""
        pass

    @abstractmethod
    def query(self, embeddings, n_results=5):
        """Query the collection with an embedding vector."""
        pass

#Chroma connector
class ChromaConnector(VectorDBConnector):
    def connect(self):
        # Use the folder for this DB type as path
        self.connection = chromadb.PersistentClient(path=self.db_dir)
        return self.connection

    def create_collection(self, overwrite=False):
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        if overwrite:
            try:
                self.connection.delete_collection(name=self.collection_name)
            except Exception:
                pass

        self.collection = self.connection.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )
        return self.collection

    def add_documents(self, ids, embeddings, metadatas):
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, embeddings, n_results=5):
        results = self.collection.query(
            query_embeddings=embeddings, n_results=n_results
        )
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]




class VectorDBCollectionsLister:
    """
    Lists ChromaDB collections for multiple database directories.
    Unsupported DBs are skipped with a warning.
    """

    def __init__(self, db_paths: List[str]):
        """
        Args:
            db_paths (List[str]): Paths to vector DB root folders (e.g. chroma_database, leann_database).
        """
        if not db_paths or not isinstance(db_paths, list):
            raise ValueError("db_paths must be a non-empty list of valid folder paths.")
        self.db_paths = db_paths

    def list_collections(self) -> Dict[str, List[str]]:
        """
        Lists all supported DB collections.

        Returns:
            Dict[str, List[str]]: Mapping of DB folder name → list of collection names.
        """
        db_collections = {}

        for db_path in self.db_paths:
            if not os.path.exists(db_path):
                warnings.warn(f"⚠️ Database folder does not exist: {db_path}")
                continue

            db_name = os.path.basename(db_path.rstrip("/\\"))

            # === Handle Chroma ===
            if "chroma" in db_name.lower():
                if chromadb is None:
                    warnings.warn("⚠️ ChromaDB not installed. Skipping Chroma database.")
                    continue

                try:
                    client = chromadb.PersistentClient(path=db_path)
                    collections = [col.name for col in client.list_collections()]
                    db_collections[db_name] = collections
                except Exception as e:
                    warnings.warn(f"⚠️ Error listing Chroma collections in '{db_name}': {e}")

            # === Fallback for unsupported DBs ===
            else:
                warnings.warn(f"⚠️ Unsupported DB type '{db_name}'. Only ChromaDB is currently supported.")
                continue

        return db_collections