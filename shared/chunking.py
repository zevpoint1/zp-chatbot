"""
chunking.py
Text chunking utilities for optimal RAG performance
"""
import re
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_by_sentences(
    text: str, 
    chunk_size: int = 500, 
    overlap: int = 50,
    min_chunk_size: int = 100
) -> List[str]:
    """
    Split text into chunks by sentences with word-based size limits
    
    Args:
        text: Input text to chunk
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
        min_chunk_size: Minimum words for a valid chunk
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split into sentences (simple regex-based)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        word_count = len(sentence.split())
        
        # If adding this sentence exceeds chunk size, save current chunk
        if current_word_count + word_count > chunk_size and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= min_chunk_size:
                chunks.append(chunk_text)
            
            # Start new chunk with overlap
            overlap_words = ' '.join(current_chunk).split()[-overlap:]
            current_chunk = [' '.join(overlap_words)] if overlap_words else []
            current_word_count = len(overlap_words)
        
        current_chunk.append(sentence)
        current_word_count += word_count
    
    # Add the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= min_chunk_size:
            chunks.append(chunk_text)
    
    logger.info(f"Created {len(chunks)} chunks from text ({len(text)} chars)")
    return chunks


def chunk_by_paragraphs(
    text: str, 
    max_chunk_size: int = 800,
    overlap_paragraphs: int = 1
) -> List[str]:
    """
    Split text by paragraphs, combining small ones
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum words per chunk
        overlap_paragraphs: Number of paragraphs to overlap
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Split by double newlines or multiple spaces
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for para in paragraphs:
        para_words = len(para.split())
        
        # If this paragraph alone exceeds max size, chunk it separately
        if para_words > max_chunk_size:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_word_count = 0
            
            # Split large paragraph by sentences
            sub_chunks = chunk_by_sentences(para, chunk_size=max_chunk_size, overlap=50)
            chunks.extend(sub_chunks)
            continue
        
        # If adding this paragraph exceeds max size, save current chunk
        if current_word_count + para_words > max_chunk_size and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
            # Keep overlap paragraphs
            if overlap_paragraphs > 0:
                current_chunk = current_chunk[-overlap_paragraphs:]
                current_word_count = sum(len(p.split()) for p in current_chunk)
            else:
                current_chunk = []
                current_word_count = 0
        
        current_chunk.append(para)
        current_word_count += para_words
    
    # Add the last chunk
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    logger.info(f"Created {len(chunks)} chunks from {len(paragraphs)} paragraphs")
    return chunks


def chunk_with_metadata(
    text: str,
    source_file: str,
    file_type: str,
    chunk_size: int = 500,
    overlap: int = 50,
    strategy: str = "sentences"
) -> List[Dict[str, Any]]:
    """
    Create chunks with rich metadata
    
    Args:
        text: Input text to chunk
        source_file: Original filename
        file_type: Type of file (pdf, docx, html, etc.)
        chunk_size: Target chunk size in words
        overlap: Overlap in words
        strategy: "sentences" or "paragraphs"
        
    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    # Choose chunking strategy
    if strategy == "paragraphs":
        chunks = chunk_by_paragraphs(text, max_chunk_size=chunk_size)
    else:
        chunks = chunk_by_sentences(text, chunk_size=chunk_size, overlap=overlap)
    
    # Add metadata to each chunk
    result = []
    for idx, chunk in enumerate(chunks):
        result.append({
            'text': chunk,
            'metadata': {
                'source_file': source_file,
                'file_type': file_type,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'char_count': len(chunk),
                'word_count': len(chunk.split())
            }
        })
    
    return result


# Test function
if __name__ == "__main__":
    test_text = """
    This is the first paragraph. It contains multiple sentences. Each sentence adds information.
    
    This is the second paragraph. It is separate from the first. We want to chunk this properly.
    
    This is the third paragraph. It should be handled correctly. The chunking algorithm should work well.
    """
    
    # Test sentence-based chunking
    chunks_sent = chunk_by_sentences(test_text, chunk_size=20, overlap=5)
    print("Sentence-based chunks:")
    for i, chunk in enumerate(chunks_sent):
        print(f"\nChunk {i+1} ({len(chunk.split())} words):")
        print(chunk[:100] + "...")
    
    # Test paragraph-based chunking
    chunks_para = chunk_by_paragraphs(test_text, max_chunk_size=50)
    print("\n\nParagraph-based chunks:")
    for i, chunk in enumerate(chunks_para):
        print(f"\nChunk {i+1} ({len(chunk.split())} words):")
        print(chunk[:100] + "...")
    
    # Test with metadata
    chunks_meta = chunk_with_metadata(
        test_text, 
        source_file="test.txt",
        file_type="txt",
        chunk_size=20
    )
    print("\n\nChunks with metadata:")
    for chunk_obj in chunks_meta:
        print(f"\nMetadata: {chunk_obj['metadata']}")
        print(f"Text preview: {chunk_obj['text'][:80]}...")