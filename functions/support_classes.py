import json
import os
import re
from typing import List, Dict
from collections import Counter
import spacy
from pathlib import Path


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
        # Replace tabs with space, normalize newlines, remove excessive markdown symbols, collapse spaces
        text = text.replace("\t", " ").replace("\n", " ")
        text = re.sub(r"[*_`~#>]+", "", text)
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

    def _build_chapters_from_splits(self, splits: List[Dict]) -> List[Dict]:
        splits = sorted(splits, key=lambda h: h["start"])
        chapters = []

        # optional preamble
        if splits and splits[0]["start"] > 0:
            pre_text = self._clean_text_for_embedding(self.text[:splits[0]["start"]].strip())
            if len(pre_text.split()) >= 20:
                chapters.append({
                    "title": "Preamble",
                    "level": None,
                    "start": 0,
                    "end": splits[0]["start"],
                    "body": pre_text,
                    "word_count": len(pre_text.split())
                })

        for i, sp in enumerate(splits):
            start = sp["start"] if self.keep_headers_in_body else sp["end"]
            end = splits[i + 1]["start"] if i + 1 < len(splits) else len(self.text)
            body = self._clean_text_for_embedding(self.text[start:end].strip())
            chapters.append({
                "title": sp.get("text") or f"Chapter {i+1}",
                "level": sp.get("level"),
                "start": start,
                "end": end,
                "body": body,
                "word_count": len(body.split())
            })
        return chapters

    # -------------------------
    # Public methods
    # -------------------------
    def split_into_chapters(self):
        book_title = os.path.splitext(os.path.basename(self.md_path))[0]

        # Find markdown headers
        header_re = re.compile(r'^(?P<hash>#{1,6})\s*(?P<text>.+?)\s*$', re.MULTILINE)
        headers = [{"start": m.start(), "end": m.end(), "level": len(m.group("hash")),
                    "text": self._clean_header_text(m.group("text"))} for m in header_re.finditer(self.text)]

        # 1) Headers with "chapter"
        chapter_headers = [h for h in headers if re.search(r'\bchapter\b', h["text"], re.IGNORECASE)]
        if len(chapter_headers) >= 2:
            self.chapters = self._build_chapters_from_splits(chapter_headers)
            # add book title
            for ch in self.chapters:
                ch["book_title"] = book_title
            return self.chapters

        # 2) High-level headers fallback
        high_level_headers = [h for h in headers if h["level"] in (1, 2)]
        if len(high_level_headers) >= 2:
            self.chapters = self._build_chapters_from_splits(high_level_headers)
            for ch in self.chapters:
                ch["book_title"] = book_title
            return self.chapters

        # 3) Plain-text "CHAPTER" lines fallback
        chap_line_re = re.compile(r'^(?P<header>\s*chapter\b[^\n]*)', re.IGNORECASE | re.MULTILINE)
        chap_lines = [{"start": m.start(), "end": m.end(), "level": None,
                       "text": self._clean_header_text(m.group("header"))} for m in chap_line_re.finditer(self.text)]
        if len(chap_lines) >= 2:
            self.chapters = self._build_chapters_from_splits(chap_lines)
            for ch in self.chapters:
                ch["book_title"] = book_title
            return self.chapters

        # 4) Last fallback: split into fixed-size chunks
        words = self.text.split()
        chunk_size = 1000
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


    def extract_entities(
        self,
        entity_types: list = [
            "PERSON", "FAC", "ORG", "GPE", "LOC", "PRODUCT",
            "EVENT", "QUANTITY", "ORDINAL", "CARDINAL"
        ],
        top_n: int = None
    ):
        """
        Extract named entities from chapters with normalization, deduplication,
        and cleanup of overlaps (e.g., removing PERSON names from other lists).

        Parameters:
            entity_types (list): List of SpaCy entity labels to extract.
            top_n (int, optional): Limit number of entities per type by frequency.

        Returns:
            list: Chapters with added "entities" key.
        """
        # Validate allowed entity labels
        permitted_entities = set(self.nlp.get_pipe("ner").labels)
        valid_entity_types = [e for e in entity_types if e in permitted_entities]
        if not valid_entity_types:
            raise ValueError(f"No valid entity types provided. Permitted: {permitted_entities}")

        for ch in self.chapters:
            doc = self.nlp(ch["body"])
            results = {}

            # Step 1: Extract entities by type
            for ent_type in valid_entity_types:
                ents = [ent.text.strip() for ent in doc.ents if ent.label_ == ent_type]
                freq = Counter(ents)
                unique = []
                for name, _ in freq.most_common():
                    # Avoid partial duplicates (e.g., "Potter" when "Harry Potter" exists)
                    if not any(name in existing or existing in name for existing in unique):
                        unique.append(name)
                if top_n:
                    unique = unique[:top_n]
                results[ent_type] = unique

            # Step 2: Remove overlaps — if something is a PERSON, remove it from all others
            if "PERSON" in results:
                person_names = set(results["PERSON"])
                for ent_type in list(results.keys()):
                    if ent_type != "PERSON":
                        results[ent_type] = [
                            ent for ent in results[ent_type]
                            if ent not in person_names
                        ]

            ch["entities"] = results

        return self.chapters





    def save_chapters_as_json(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.md_path))[0]

        for i, ch in enumerate(self.chapters, 1):
            safe_title = self._make_safe_filename(ch["title"])
            filename = f"{i:02d} - {base_name} - {safe_title}.json"
            filepath = os.path.join(output_dir, filename)

            # Save the full chapter dict as JSON
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(ch, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(self.chapters)} chapters as JSON to {output_dir}")


