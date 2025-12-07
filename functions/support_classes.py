# Initialization + default paths
from utilities.paths import *

# Settings 
from settings.transformation_settings import *
from settings.vector_db_settings import DEFAULT_DB_TYPES,COLLECTION_META_DATA

# General Imports
import json
import os
import re
from typing import List, Dict,Optional,Any
from collections import Counter
from pathlib import Path
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timezone

# NLP Import 
import spacy

# ChromaDB imports
import chromadb
from chromadb.utils import embedding_functions
import inspect
from sentence_transformers import SentenceTransformer
import numpy as np



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

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
        func_name = inspect.currentframe().f_code.co_name

        self.data = []
        if not os.path.exists(file_path):
            # Create empty file
            with open(file_path, "w", encoding="utf-8") as f:
                pass
            print(f"{func_name}: File '{file_path}' did not exist and was created as an empty JSONL file.")
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
        func_name = inspect.currentframe().f_code.co_name
        if cls._shared_nlp is None:
            import spacy
            print(f"{func_name}: Loading spaCy model (once)...")
            cls._shared_nlp = spacy.load("en_core_web_sm")
        return cls._shared_nlp

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

        func_name = inspect.currentframe().f_code.co_name
        # Validate allowed entity labels
        permitted_entities = set(self.nlp.get_pipe("ner").labels)
        valid_entity_types = [e for e in entity_types if e in permitted_entities]
        if not valid_entity_types:
            raise ValueError(f"{func_name}: No valid entity types provided. Permitted: {permitted_entities}")

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
        func_name = inspect.currentframe().f_code.co_name
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.md_path))[0]
       
        for i, ch in enumerate(self.chapters, 1):
            
            safe_title = self._make_safe_filename(ch["book_title"])
            filename = f"{i:02d} - {base_name} - {safe_title}.json"
            filepath = os.path.join(output_dir, filename)

            # Save the full chapter dict as JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(ch, f, ensure_ascii=False, indent=2)

        print(f"{func_name}: Saved {len(self.chapters)} chapters as JSON to {output_dir}")


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

    
    def __init__(self,*, db_dir:str, collection_name: str,collection_metadata:dict = None, db_type: str = "chroma", overwrite: bool = False, **kwargs):
        """
        Base constructor for vector DB connectors.
        Args:
            collection_name (str): Name of the collection.
            db_type (str): Type of vector DB ('chroma' or 'leann').
            overwrite (bool): Whether to overwrite an existing collection.
            kwargs: DB-specific parameters.
        """
        self.collection_name = collection_name
        
        self.kwargs = kwargs
        self.connection = None
        self.collection = None
        self.db_dir = db_dir

        


        if collection_metadata is not None:
            now = datetime.now(timezone.utc).date().isoformat()
            # Ensure all required keys exist, updating date fields automatically
            self.collection_metadata = {
                "description": collection_metadata.get("description", ""),
                "creation_date": now,
                "collection_data_last_updated": now,
                "collection_metadata_last_updated": now,
            }
        else:
            now = datetime.now(timezone.utc).date().isoformat()
            self.collection_metadata = {
                "description": "",
                "creation_date": now,
                "collection_data_last_updated": "",
                "collection_metadata_last_updated": "",
            }

        self.db_type = db_type.lower()
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

    @abstractmethod
    def update_collection_metadata(self, meta_data):
        """update the collection metadata"""
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

        self.collection = self.connection.get_or_create_collection(name=self.collection_name, metadata = self.collection_metadata)
        return self.collection

    def add_documents(self, ids, embeddings, metadatas):
        try:
            print("adding collection from ChromaConnector class")
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            print("Failed to add document to Chroma")
            print(f"IDs: {ids}")
            print(f"Metadatas: {metadatas}")
            print(f"Error: {e}")

    def query_old(self, embeddings, n_results=5):
        results = self.collection.query(query_embeddings=embeddings, n_results=n_results)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
    
    def query(self, embeddings, n_results = 10, where=None,retrieval_list: list | None = ["documents", "metadatas"],  include_distance: bool = True):
        
    
 
        if include_distance:
            retrieval_list.append("distances")
            
        # Stage 1. Get raw results from ChromaDB
        results = self.collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
            include = retrieval_list
        )
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        ids = results["ids"][0]
        dists = results.get("distances", [[None]])[0] if include_distance else [None] * len(ids)
        
        items = [
            {"id": i, "document": d, "metadata": m, "cosine_distance": dist,}
            for i, d, m, dist in zip(ids, docs, metas,dists)
        ]
        

        # Stage 2. Local filtering based on string search
        if where:
            filtered = []
            for item in items:
                text = (item["document"] or "").lower()
                meta_text = " ".join(str(v) for v in item["metadata"].values()).lower()
                combined = text + " " + meta_text

                if any(t.lower() in combined for t in where):
                    filtered.append(item)

            return filtered

        return items
    
    def update_collection_metadata(self, meta_data: dict):
        """Update the collection metadata, ignoring creation_date and None values."""
        
        
        filtered_meta = {k: v for k, v in meta_data.items() if v is not None and k in self.collection_metadata}
        self.collection_metadata["collection_metadata_last_updated"] = datetime.now(timezone.utc).date().isoformat()

        self.collection_metadata.update(filtered_meta)

        self.collection.modify(metadata=self.collection_metadata)

        
        
    def delete_by_metadata(self, filters: dict):
        """
        Delete all entries that match the given metadata filters.
        Example: filters={"source": "pdf1"} will delete all items
        where metadata["source"] equals "pdf1".
        """

        if not isinstance(filters, dict):
            raise ValueError("Filters must be a dictionary of metadata keys and values")

        try:
            self.collection.delete(where=filters)
            return True
        except Exception as exc:
            return False
    
    def get_collection_data(self):
        """
        Return every record in the collection except embeddings.
        Only ids, documents, and metadata are returned.
        """
        
        # excludes "embeddings"
        results = self.collection.get(include=["documents", "metadatas"])       

        items = []
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        items = [{"id": i, "document": d, "metadata": m} for i, d, m in zip(ids, docs, metas)]

        return items
    
    def delete_collection(self, collection_name:str):
        """
        Delete the entire collection from the vector store.
        This removes all documents and metadata for this collection.
        """
        try:
            self.connection.delete_collection(name = collection_name)
            return True
        except Exception as exc:
            print(f"Failed to delete collection: {exc}")
            return False




class VectorDBCollectionsEditor:
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
    

class EmbeddingLLM_Manager:
    def __init__(self, model_name: str) -> None:
        
        func_name = inspect.currentframe().f_code.co_name
        if model_name is None or model_name == "":
            raise ValueError(f"{func_name}: model_name cannot be empty")
            
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        
        self.model = SentenceTransformer(self.model_name, device = self._get_device())

    def encode_prompt(self, prompt: str) -> np.ndarray:
        return self.model.encode(prompt)
    
    def get_model(self) -> SentenceTransformer:
        return self.model
    
    def _get_device(self) -> str:
        try:
            import torch
        except ImportError:
            return "cpu"

        return "cuda" if torch.cuda.is_available() else "cpu"


class LLMManager:
    def __init__(self, model_name: str, mode: str) -> None:
        func_name = inspect.currentframe().f_code.co_name

        if not model_name:
            raise ValueError(f"{func_name}: model_name cannot be empty")

        if mode not in ("encoder", "decoder"):
            raise ValueError(f"{func_name}: mode must be 'encoder' or 'decoder'")

        self.model_name = model_name
        self.mode = mode
        self.device = self._get_device()
        self._load_model()

    @staticmethod
    def _get_device() -> str:
        try:
            import torch  # noqa
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_model(self) -> None:
        if self.mode == "encoder":
            self.model = SentenceTransformer(self.model_name, device=self.device)

        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device
            )

            gen_pipeline = pipeline(
                task="text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7
            )

            self.tokenizer = tokenizer
            self.model = model
            self.llm = HuggingFacePipeline(pipeline=gen_pipeline)

    def encode(self, prompt: str | np.ndarray) -> np.ndarray:
        func_name = inspect.currentframe().f_code.co_name

        if self.mode != "encoder":
            raise RuntimeError(f"{func_name}: encode called in decoder mode")

        if isinstance(prompt, str):
            return self.model.encode(prompt)

        if isinstance(prompt, np.ndarray):
            return prompt

        raise TypeError(f"{func_name}: prompt must be str or np.ndarray")

    def generate(self, prompt: str) -> str:
        func_name = inspect.currentframe().f_code.co_name
        if self.mode != "decoder":
            raise RuntimeError(f"{func_name}: generate called in encoder mode")
        return self.llm.invoke(prompt)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        func_name = inspect.currentframe().f_code.co_name
        if self.mode != "decoder":
            raise RuntimeError(f"{func_name}: Tokenizer is only available in decoder mode")
        return self.tokenizer

    def get_langchain_llm(self):
        func_name = inspect.currentframe().f_code.co_name
        if self.mode != "decoder":
            raise RuntimeError(f"{func_name}: Langchain interface is only available in decoder mode")
        return self.llm


class Document_Finder:
    def __init__(self, model_name:str, vectorDBManager) -> None:
        
        self._load_or_create_model(model_name = model_name)
        self.vectorDBManager = vectorDBManager
    
    def _load_or_create_model(self, model_name:str) -> None:
        
        func_name = inspect.currentframe().f_code.co_name
        
        if isinstance(model_name, str):
            self.embeddingLLM_manager = LLMManager(model_name = model_name, mode = "encoder")
            
        else:
            raise ValueError(f"{func_name}: model can either be model name (str) or loaded model (SentenceTransformer)")
        
    def _prepare_query(self, query:str|np.ndarray) -> np.ndarray:
        
        return self.embeddingLLM_manager.encode(query)
    
    def query_documents(self, query:str|np.ndarray, n_results:int = 50, where:Optional[dict] = None) -> list[dict]:
        
        func_name = inspect.currentframe().f_code.co_name
        
        embeddings = self._prepare_query(query)

        return self.vectorDBManager.query(
            embeddings=embeddings,
            n_results=n_results,
            where=where,
        )