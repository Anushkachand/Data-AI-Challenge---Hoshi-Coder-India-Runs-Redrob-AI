"""
io_utils.py
Input/Output utilities for the Redrob candidate ranker.
Handles candidate JSONL streaming, Word docx extraction, FAISS, and NumPy IO.
Follows PEP 8 style guide.
"""

import gzip
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# Try to import faiss, but do not crash immediately if it is not installed yet
try:
    import faiss
except ImportError:
    faiss = None


def stream_candidates(file_path):
    """
    Stream candidates line-by-line from candidates.jsonl or candidates.jsonl.gz.
    Yields candidate dictionaries.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    # Check if the file is gzipped
    if path.suffix == ".gz" or path.name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def load_docx_text(docx_path):
    """
    Reads a Word .docx document and returns its full text.
    Uses python-docx if available, otherwise falls back to zipfile/xml parsing.
    """
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"docx file not found: {path}")

    # Try using python-docx first
    try:
        import docx
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        # Fallback to direct zip/xml extraction
        try:
            with zipfile.ZipFile(path) as docx_zip:
                xml_content = docx_zip.read('word/document.xml')
                root = ET.fromstring(xml_content)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                paragraphs = []
                for p in root.findall('.//w:p', ns):
                    text_runs = []
                    for t in p.findall('.//w:t', ns):
                        if t.text:
                            text_runs.append(t.text)
                    if text_runs:
                        paragraphs.append(''.join(text_runs))
                return '\n'.join(paragraphs)
        except Exception as e:
            raise IOError(f"Failed to read docx file {path}: {e}")


def save_numpy_array(arr, path):
    """
    Saves a NumPy array to the specified path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def load_numpy_array(path):
    """
    Loads a NumPy array from the specified path.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"NumPy array not found at: {path}")
    return np.load(path)


def save_faiss_index(index, path):
    """
    Saves a FAISS index to the specified path.
    """
    if faiss is None:
        raise ImportError("FAISS is not installed.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path):
    """
    Loads a FAISS index from the specified path.
    """
    if faiss is None:
        raise ImportError("FAISS is not installed.")
    if not Path(path).exists():
        raise FileNotFoundError(f"FAISS index not found at: {path}")
    return faiss.read_index(str(path))
