"""
ingest.py
Document ingestion pipeline for RAG system
Supports: PDF, DOCX, TXT, HTML, Images (OCR), URL ingestion, JSON, JSONL
Includes: Deduplication + Batch Embedding + Clean Qdrant Upload + Enhanced JSON Context
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import requests
from requests.adapters import HTTPAdapter

# Extractors
from pypdf import PdfReader
import docx
from PIL import Image
import pytesseract
from bs4 import BeautifulSoup

# Local modules
from shared.embed_text import embed_batch
from shared.chunking import chunk_with_metadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qdrant config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ev_kb")

# Tesseract path (Windows)
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# HTTP session with connection pooling
_http_session = None

def get_http_session():
    """Get or create HTTP session with connection pooling for Qdrant"""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=3
        )
        _http_session.mount('http://', adapter)
        _http_session.mount('https://', adapter)
        logger.info("HTTP session initialized for ingestion")
    return _http_session


# -------------------------------------------------------------------
# 🔹 Extractors
# -------------------------------------------------------------------

def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()
    except Exception as e:
        logger.error(f"PDF extraction failed: {e}")
        return ""


def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        return text.strip()
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return ""


def extract_text_from_txt(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"TXT extraction failed: {e}")
        return ""


def extract_text_from_html(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    except Exception as e:
        logger.error(f"HTML extraction failed: {e}")
        return ""


def extract_text_from_url(url: str) -> str:
    """Extract readable text from a URL."""
    try:
        logger.info(f"Downloading URL: {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])

        logger.info(f"Extracted {len(text)} chars from URL: {url}")
        return text.strip()

    except Exception as e:
        logger.error(f"URL extraction error: {e}")
        return ""


def extract_text_from_image(file_path):
    try:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return ""


def extract_text_from_json(file_path: str) -> str:
    """
    Extract text from JSON with enhanced semantic context.
    Supports FAQ, product catalogs, and generic JSON structures.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        filename = os.path.basename(file_path)
        text_parts = [f"Source Document: {filename}\n"]
        
        # Handle JSON array (list of items)
        if isinstance(data, list):
            text_parts.append(f"This document contains {len(data)} entries.\n")
            
            for i, item in enumerate(data, 1):
                if not isinstance(item, dict):
                    if isinstance(item, str):
                        text_parts.append(f"Entry {i}: {item}")
                    continue
                
                # Detect structure type
                is_faq = 'question' in item and 'answer' in item
                is_product = any(k in item for k in ['product', 'title', 'name'])
                
                text_parts.append(f"\n{'='*50}")
                text_parts.append(f"Entry {i}")
                text_parts.append('='*50)
                
                if is_faq:
                    # FAQ format
                    text_parts.append(f"\nQuestion: {item.get('question', '')}")
                    text_parts.append(f"Answer: {item.get('answer', '')}")
                    
                    if 'category' in item:
                        text_parts.append(f"Category: {item['category']}")
                    
                    # Add any other fields
                    for key, value in item.items():
                        if key not in ['question', 'answer', 'category']:
                            readable_key = key.replace("_", " ").replace("-", " ").title()
                            if isinstance(value, (str, int, float)):
                                text_parts.append(f"{readable_key}: {value}")
                
                elif is_product:
                    # Product format
                    product_name = item.get('product') or item.get('title') or item.get('name', f'Product {i}')
                    text_parts.append(f"\nProduct Name: {product_name}")
                    
                    if 'description' in item:
                        text_parts.append(f"Description: {item['description']}")
                    
                    # Handle specifications
                    if 'specs' in item or 'specifications' in item:
                        specs = item.get('specs') or item.get('specifications')
                        text_parts.append("\nSpecifications:")
                        if isinstance(specs, dict):
                            for key, value in specs.items():
                                readable_key = key.replace("_", " ").replace("-", " ").title()
                                text_parts.append(f"  • {readable_key}: {value}")
                        else:
                            text_parts.append(f"  {specs}")
                    
                    # Handle features
                    if 'features' in item:
                        features = item['features']
                        text_parts.append("\nKey Features:")
                        if isinstance(features, list):
                            for feature in features:
                                text_parts.append(f"  • {feature}")
                        else:
                            text_parts.append(f"  {features}")
                    
                    # Handle pricing
                    if 'price' in item:
                        text_parts.append(f"\nPrice: {item['price']}")
                    
                    # Handle warranty
                    if 'warranty' in item:
                        text_parts.append(f"Warranty: {item['warranty']}")
                    
                    # Handle other common fields
                    for key in ['model', 'sku', 'brand', 'availability', 'rating']:
                        if key in item:
                            readable_key = key.replace("_", " ").replace("-", " ").title()
                            text_parts.append(f"{readable_key}: {item[key]}")
                    
                    # Add remaining fields (excluding already processed ones)
                    processed_keys = {
                        'product', 'title', 'name', 'description', 'specs', 
                        'specifications', 'features', 'price', 'warranty',
                        'model', 'sku', 'brand', 'availability', 'rating'
                    }
                    
                    remaining = {k: v for k, v in item.items() if k not in processed_keys}
                    if remaining:
                        text_parts.append("\nAdditional Information:")
                        for key, value in remaining.items():
                            readable_key = key.replace("_", " ").replace("-", " ").title()
                            if isinstance(value, dict):
                                text_parts.append(f"{readable_key}:")
                                for sub_key, sub_value in value.items():
                                    sub_readable = sub_key.replace("_", " ").replace("-", " ").title()
                                    text_parts.append(f"  • {sub_readable}: {sub_value}")
                            elif isinstance(value, list):
                                text_parts.append(f"{readable_key}: {', '.join(map(str, value))}")
                            else:
                                text_parts.append(f"{readable_key}: {value}")
                
                else:
                    # Generic object - extract all fields with readable formatting
                    for key, value in item.items():
                        readable_key = key.replace("_", " ").replace("-", " ").title()
                        
                        if isinstance(value, dict):
                            text_parts.append(f"\n{readable_key}:")
                            for sub_key, sub_value in value.items():
                                sub_readable = sub_key.replace("_", " ").replace("-", " ").title()
                                text_parts.append(f"  • {sub_readable}: {sub_value}")
                        
                        elif isinstance(value, list):
                            text_parts.append(f"\n{readable_key}:")
                            for item_val in value:
                                text_parts.append(f"  • {item_val}")
                        
                        elif isinstance(value, (str, int, float)) and value:
                            text_parts.append(f"{readable_key}: {value}")
        
        # Handle JSON object (single dictionary or nested structure)
        elif isinstance(data, dict):
            # Check if it's a wrapper object containing a list
            list_keys = ['products', 'items', 'entries', 'data', 'records', 'faqs']
            found_list = None
            
            for key in list_keys:
                if key in data and isinstance(data[key], list):
                    found_list = key
                    break
            
            if found_list:
                # Recursively process the list
                text_parts.append(f"\nDocument Type: {found_list.replace('_', ' ').title()}")
                text_parts.append(f"Total Count: {len(data[found_list])}\n")
                
                # Process each item in the list
                for i, item in enumerate(data[found_list], 1):
                    if not isinstance(item, dict):
                        continue
                    
                    text_parts.append(f"\n{'='*50}")
                    text_parts.append(f"Item {i}")
                    text_parts.append('='*50)
                    
                    # Use similar logic as array handling
                    for key, value in item.items():
                        readable_key = key.replace("_", " ").replace("-", " ").title()
                        
                        if isinstance(value, dict):
                            text_parts.append(f"\n{readable_key}:")
                            for sub_key, sub_value in value.items():
                                sub_readable = sub_key.replace("_", " ").replace("-", " ").title()
                                text_parts.append(f"  • {sub_readable}: {sub_value}")
                        
                        elif isinstance(value, list):
                            text_parts.append(f"\n{readable_key}:")
                            for val in value:
                                text_parts.append(f"  • {val}")
                        
                        elif isinstance(value, (str, int, float)) and value:
                            text_parts.append(f"{readable_key}: {value}")
            
            else:
                # Single object - extract recursively
                text_parts.append("Document Information:\n")
                
                def extract_nested(obj, level=0):
                    """Recursively extract nested structures"""
                    indent = "  " * level
                    
                    for key, value in obj.items():
                        readable_key = key.replace("_", " ").replace("-", " ").title()
                        
                        if isinstance(value, dict):
                            text_parts.append(f"{indent}{readable_key}:")
                            extract_nested(value, level + 1)
                        
                        elif isinstance(value, list):
                            text_parts.append(f"{indent}{readable_key}:")
                            for idx, item in enumerate(value):
                                if isinstance(item, dict):
                                    text_parts.append(f"{indent}  Item {idx + 1}:")
                                    extract_nested(item, level + 2)
                                else:
                                    text_parts.append(f"{indent}  • {item}")
                        
                        elif isinstance(value, (str, int, float, bool)) and value is not None:
                            text_parts.append(f"{indent}{readable_key}: {value}")
                
                extract_nested(data)
        
        else:
            # Fallback for primitive types
            text_parts.append(f"Content: {str(data)}")
        
        result = "\n".join(text_parts)
        logger.info(f"Extracted {len(result)} characters from JSON file")
        return result
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in {file_path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""


def extract_text_from_jsonl(file_path: str) -> str:
    """
    Extract text from JSONL (newline-delimited JSON) with enhanced formatting.
    Each line is a separate JSON object.
    """
    try:
        filename = os.path.basename(file_path)
        text_parts = [f"Source Document: {filename}"]
        text_parts.append("Format: JSONL (Newline-delimited JSON)\n")
        
        entry_count = 0
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    entry_count += 1
                    
                    text_parts.append(f"\n{'='*50}")
                    text_parts.append(f"Record {entry_count} (Line {line_num})")
                    text_parts.append('='*50)
                    
                    # Extract based on structure
                    if isinstance(data, dict):
                        # Check for FAQ format
                        if 'question' in data and 'answer' in data:
                            text_parts.append(f"\nQuestion: {data['question']}")
                            text_parts.append(f"Answer: {data['answer']}")
                            
                            # Add other fields
                            for key, value in data.items():
                                if key not in ['question', 'answer']:
                                    readable_key = key.replace("_", " ").replace("-", " ").title()
                                    text_parts.append(f"{readable_key}: {value}")
                        
                        else:
                            # Generic dictionary
                            for key, value in data.items():
                                readable_key = key.replace("_", " ").replace("-", " ").title()
                                
                                if isinstance(value, dict):
                                    text_parts.append(f"\n{readable_key}:")
                                    for sub_key, sub_value in value.items():
                                        sub_readable = sub_key.replace("_", " ").replace("-", " ").title()
                                        text_parts.append(f"  • {sub_readable}: {sub_value}")
                                
                                elif isinstance(value, list):
                                    text_parts.append(f"{readable_key}:")
                                    for item in value:
                                        text_parts.append(f"  • {item}")
                                
                                elif isinstance(value, (str, int, float)) and value:
                                    text_parts.append(f"{readable_key}: {value}")
                    
                    elif isinstance(data, str):
                        text_parts.append(f"Content: {data}")
                    
                    else:
                        text_parts.append(f"Data: {str(data)}")
                
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON on line {line_num} in {file_path}")
                    continue
        
        text_parts.insert(2, f"Total Records: {entry_count}\n")
        
        result = "\n".join(text_parts)
        logger.info(f"Extracted {entry_count} records from JSONL file")
        return result
    
    except Exception as e:
        logger.error(f"JSONL extraction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return ""


def extract_text(file_path: str) -> str:
    """Main extraction router based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".doc": extract_text_from_docx,
        ".txt": extract_text_from_txt,
        ".html": extract_text_from_html,
        ".htm": extract_text_from_html,
        ".png": extract_text_from_image,
        ".jpg": extract_text_from_image,
        ".jpeg": extract_text_from_image,
        ".gif": extract_text_from_image,
        ".bmp": extract_text_from_image,
        ".json": extract_text_from_json,
        ".jsonl": extract_text_from_jsonl,
    }

    extractor = extractors.get(ext)
    if not extractor:
        logger.warning(f"Unsupported file type: {ext}")
        return ""

    return extractor(file_path)


# -------------------------------------------------------------------
# 🔹 JSON Structure Detection
# -------------------------------------------------------------------

def detect_json_structure(file_path: str) -> Dict[str, Any]:
    """
    Detect the structure and content type of a JSON file.
    Returns metadata about the JSON content.
    """
    metadata = {
        'content_type': 'unknown',
        'item_count': 0,
        'has_questions': False,
        'has_products': False,
        'structure': 'unknown'
    }
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            metadata['structure'] = 'array'
            metadata['item_count'] = len(data)
            
            if len(data) > 0 and isinstance(data[0], dict):
                first_item = data[0]
                
                # Detect FAQ
                if 'question' in first_item and 'answer' in first_item:
                    metadata['content_type'] = 'faq'
                    metadata['has_questions'] = True
                
                # Detect product catalog
                elif any(k in first_item for k in ['product', 'title', 'name', 'price']):
                    metadata['content_type'] = 'product_catalog'
                    metadata['has_products'] = True
                
                # Detect specifications
                elif 'specs' in first_item or 'specifications' in first_item:
                    metadata['content_type'] = 'specifications'
        
        elif isinstance(data, dict):
            # Check for wrapper objects
            list_keys = ['products', 'items', 'entries', 'faqs', 'data']
            for key in list_keys:
                if key in data and isinstance(data[key], list):
                    metadata['structure'] = 'wrapped_array'
                    metadata['item_count'] = len(data[key])
                    
                    if key == 'faqs':
                        metadata['content_type'] = 'faq'
                        metadata['has_questions'] = True
                    elif key == 'products':
                        metadata['content_type'] = 'product_catalog'
                        metadata['has_products'] = True
                    
                    break
            else:
                metadata['structure'] = 'object'
        
        logger.info(f"Detected JSON structure: {metadata}")
        return metadata
    
    except Exception as e:
        logger.error(f"Error detecting JSON structure: {e}")
        return metadata


# -------------------------------------------------------------------
# 🔹 Deduplication Logic
# -------------------------------------------------------------------

def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_existing_hashes() -> set:
    """Retrieve existing text hashes from Qdrant to avoid duplicates with pagination."""
    url = f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}

    existing = set()
    offset = None
    session = get_http_session()

    try:
        while True:
            payload = {
                "with_payload": ["text_hash"],  # Only fetch text_hash field for efficiency
                "limit": 1000,  # Process in batches
            }

            if offset:
                payload["offset"] = offset

            resp = session.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            points = result.get("result", {}).get("points", [])
            if not points:
                break

            for p in points:
                h = p.get("payload", {}).get("text_hash")
                if h:
                    existing.add(h)

            # Check for next page
            next_offset = result.get("result", {}).get("next_page_offset")
            if next_offset is None:
                break

            offset = next_offset

        logger.info(f"Found {len(existing)} existing hashes in Qdrant.")

    except Exception as e:
        logger.error(f"Error loading existing hashes: {e}")

    return existing


# -------------------------------------------------------------------
# 🔹 Upload chunks to Qdrant
# -------------------------------------------------------------------

def upload_chunks(chunks: List[Dict[str, Any]]) -> bool:
    """Upload chunks to Qdrant with deduplication."""
    if not chunks:
        logger.warning("No chunks to upload.")
        return False

    existing_hashes = fetch_existing_hashes()

    # Filter duplicates
    new_chunks = []
    duplicate_count = 0
    
    for chunk in chunks:
        text = chunk["text"]
        h = compute_hash(text)

        if h in existing_hashes:
            duplicate_count += 1
            logger.debug(f"Skipping duplicate chunk (hash: {h[:12]}...)")
            continue

        chunk["metadata"]["text_hash"] = h
        new_chunks.append(chunk)

    if duplicate_count > 0:
        logger.info(f"Filtered out {duplicate_count} duplicate chunks")

    if not new_chunks:
        logger.info("All chunks already existed. Nothing to upload.")
        return True

    logger.info(f"Preparing to upload {len(new_chunks)} new chunks")

    # Batch embed with size limits
    MAX_BATCH_SIZE = 500  # Limit to avoid API rate limits and memory issues

    if len(new_chunks) > MAX_BATCH_SIZE:
        logger.warning(f"Chunk count ({len(new_chunks)}) exceeds max batch size ({MAX_BATCH_SIZE}). Processing in batches.")

    try:
        all_vectors = []

        # Process in batches to respect API limits
        for batch_start in range(0, len(new_chunks), MAX_BATCH_SIZE):
            batch_end = min(batch_start + MAX_BATCH_SIZE, len(new_chunks))
            batch_chunks = new_chunks[batch_start:batch_end]

            logger.info(f"Processing embedding batch {batch_start//MAX_BATCH_SIZE + 1}: chunks {batch_start+1}-{batch_end}")

            texts = [c["text"] for c in batch_chunks]
            vectors = embed_batch(texts, batch_size=20)  # Embed in sub-batches of 20

            all_vectors.extend(vectors)

        if len(all_vectors) != len(new_chunks):
            logger.error(f"Vector count mismatch: {len(all_vectors)} vectors for {len(new_chunks)} chunks")
            return False

        logger.info(f"Generated embeddings for {len(all_vectors)} chunks")
        vectors = all_vectors

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

    # Prepare Qdrant points
    points = []
    timestamp = datetime.now()

    for i, (chunk, vec) in enumerate(zip(new_chunks, vectors)):
        # Generate unique ID using timestamp and index
        point_id = int(timestamp.timestamp() * 1_000_000) + i

        points.append({
            "id": point_id,
            "vector": vec,  # ✅ Use "vector" for unnamed vector collections
            "payload": {
                "text": chunk["text"],
                **chunk["metadata"],
                "ingested_at": timestamp.isoformat()
            }
        })

    # Upload to Qdrant using session
    url = f"{QDRANT_URL}/collections/{COLLECTION}/points"
    headers = {"api-key": QDRANT_API_KEY, "Content-Type": "application/json"}
    session = get_http_session()

    try:
        resp = session.put(
            url,
            headers=headers,
            json={"points": points},
            timeout=60  # Increased timeout for large batches
        )
        resp.raise_for_status()
        
        logger.info(f"✅ Successfully uploaded {len(points)} new chunks to Qdrant")
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Qdrant upload failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response: {e.response.text}")
        return False


# -------------------------------------------------------------------
# 🔹 Main ingestion function
# -------------------------------------------------------------------

def ingest(source: str, chunk_size: int = 300, overlap: int = 75) -> bool:
    """
    Ingest local files or URLs into the RAG system.
    
    Args:
        source: File path or URL
        chunk_size: Target chunk size in words
        overlap: Overlap between chunks in words
    
    Returns:
        bool: True if ingestion successful, False otherwise
    """
    logger.info(f"Starting ingestion: {source}")
    
    is_url = source.startswith("http://") or source.startswith("https://")

    # Extract text
    if is_url:
        text = extract_text_from_url(source)
        filename = source
        file_type = "url"
        content_metadata = {}
    else:
        if not os.path.exists(source):
            logger.error(f"❌ File not found: {source}")
            return False

        filename = os.path.basename(source)
        file_extension = os.path.splitext(filename)[1].lower()
        file_type = file_extension.lstrip('.')
        
        # Detect JSON structure for metadata
        content_metadata = {}
        if file_extension == '.json':
            content_metadata = detect_json_structure(source)
        
        text = extract_text(source)

    if not text or not text.strip():
        logger.warning("⚠️  No extractable text found. Skipping.")
        return False

    logger.info(f"Extracted {len(text)} characters from {filename}")

    # Chunk the text
    try:
        chunks = chunk_with_metadata(
            text=text,
            source_file=filename,
            file_type=file_type,
            strategy="sentences",
            chunk_size=chunk_size,
            overlap=overlap
        )
        logger.info(f"Created {len(chunks)} chunks from document")
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        return False

    # Enrich chunks with detected metadata
    for chunk in chunks:
        chunk['metadata'].update(content_metadata)
    
    # Upload chunks
    success = upload_chunks(chunks)
    
    if success:
        logger.info(f"✅ Ingestion completed successfully: {filename}")
    else:
        logger.error(f"❌ Ingestion failed: {filename}")
    
    return success


# -------------------------------------------------------------------
# 🔹 Batch Ingestion
# -------------------------------------------------------------------

def ingest_directory(directory: str, pattern: str = "*", chunk_size: int = 300, overlap: int = 75) -> Dict[str, Any]:
    """
    Ingest all matching files from a directory.
    
    Args:
        directory: Path to directory
        pattern: File pattern (e.g., "*.json", "*.pdf")
        chunk_size: Target chunk size
        overlap: Overlap between chunks
    
    Returns:
        dict: Summary of ingestion results
    """
    import glob
    
    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return {"error": "Directory not found"}
    
    # Find matching files
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    
    if not files:
        logger.warning(f"No files matching pattern '{pattern}' found in {directory}")
        return {"processed": 0, "successful": 0, "failed": 0, "files": []}
    
    logger.info(f"Found {len(files)} files to process")
    
    results = {
        "processed": 0,
        "successful": 0,
        "failed": 0,
        "files": []
    }
    
    for file_path in files:
        results["processed"] += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing file {results['processed']}/{len(files)}: {file_path}")
        logger.info('='*60)
        
        try:
            success = ingest(file_path, chunk_size=chunk_size, overlap=overlap)
            
            if success:
                results["successful"] += 1
                results["files"].append({"file": file_path, "status": "success"})
            else:
                results["failed"] += 1
                results["files"].append({"file": file_path, "status": "failed"})
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            results["failed"] += 1
            results["files"].append({"file": file_path, "status": "error", "error": str(e)})
    
    logger.info(f"\n{'='*60}")
    logger.info("BATCH INGESTION SUMMARY")
    logger.info('='*60)
    logger.info(f"Total files processed: {results['processed']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info('='*60)
    
    return results


# -------------------------------------------------------------------
# 🔹 CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest documents into RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest a single file
  python ingest.py document.pdf
  
  # Ingest a JSON file
  python ingest.py products.json
  
  # Ingest from URL
  python ingest.py https://example.com/article
  
  # Ingest all JSON files from a directory
  python ingest.py --directory ./data --pattern "*.json"
  
  # Custom chunk size
  python ingest.py document.pdf --chunk-size 500 --overlap 100
        """
    )
    
    parser.add_argument(
        "source",
        nargs="?",
        help="File path or URL to ingest"
    )
    
    parser.add_argument(
        "--directory", "-d",
        help="Ingest all files from directory"
    )
    
    parser.add_argument(
        "--pattern", "-p",
        default="*",
        help="File pattern for directory ingestion (default: *)"
    )
    
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=300,
        help="Target chunk size in words (default: 300)"
    )
    
    parser.add_argument(
        "--overlap", "-o",
        type=int,
        default=75,
        help="Overlap between chunks in words (default: 75)"
    )
    
    args = parser.parse_args()

    # Validate environment variables
    if not all([QDRANT_URL, QDRANT_API_KEY, COLLECTION]):
        logger.error("Missing required environment variables:")
        logger.error("  - QDRANT_URL")
        logger.error("  - QDRANT_API_KEY")
        logger.error("  - QDRANT_COLLECTION")
        sys.exit(1)

    # Process directory or single file
    if args.directory:
        results = ingest_directory(
            args.directory,
            pattern=args.pattern,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        sys.exit(0 if results["failed"] == 0 else 1)
    
    elif args.source:
        success = ingest(
            args.source,
            chunk_size=args.chunk_size,
            overlap=args.overlap
        )
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        sys.exit(1)