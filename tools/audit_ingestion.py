import os
import sys
import json
from pathlib import Path
from typing import Dict, List

# ensure repo root is on path for imports
sys.path.insert(0, os.getcwd())

from shared.chunking import chunk_with_metadata
from upload_to_qdrant import fetch_existing_hashes, text_hash

UPLOADS_DIR = Path(os.getenv('UPLOADS_DIR', 'uploads'))


def chunks_for_file(path: Path) -> List[Dict]:
    text = ''
    if path.suffix.lower() in ['.txt', '.md']:
        text = path.read_text(encoding='utf-8')
        chunks = chunk_with_metadata(text, source_file=path.name, file_type='txt')
    elif path.suffix.lower() in ['.jsonl']:
        chunks = []
        for i, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                t = obj.get('text') or obj.get('content') or json.dumps(obj)
            except Exception:
                t = line
            sub_chunks = chunk_with_metadata(t, source_file=path.name, file_type='jsonl')
            # attach original line index
            for sc in sub_chunks:
                sc['metadata']['line_index'] = i
            chunks.extend(sub_chunks)
    else:
        # Fallback: read as text and chunk
        text = path.read_text(encoding='utf-8', errors='ignore')
        chunks = chunk_with_metadata(text, source_file=path.name, file_type=path.suffix.lstrip('.'))

    # compute hash per chunk
    for c in chunks:
        c['chunk_hash'] = text_hash(c['text'])
    return chunks


if __name__ == '__main__':
    print('Scanning local upload files in', UPLOADS_DIR)
    files = list(UPLOADS_DIR.glob('**/*'))
    files = [f for f in files if f.is_file()]

    # Gather local chunk hashes
    local_index = {}  # file -> list of hashes
    total_chunks = 0
    for f in files:
        try:
            ch = chunks_for_file(f)
            hashes = [c['chunk_hash'] for c in ch]
            total_chunks += len(hashes)
            local_index[str(f)] = hashes
            print(f'  {f.name}: {len(hashes)} chunks')
        except Exception as e:
            print(f'Failed to process {f}: {e}')

    print(f'Total local chunks: {total_chunks}')

    # Fetch existing hashes from Qdrant
    print('Fetching existing chunk hashes from Qdrant...')
    existing = fetch_existing_hashes()
    print(f'Found {len(existing)} existing hashes in Qdrant')

    # Compare
    missing_by_file = {}
    present = 0
    missing = 0
    for fname, hashes in local_index.items():
        missing_hashes = [h for h in hashes if h not in existing]
        missing_by_file[fname] = missing_hashes
        missing += len(missing_hashes)
        present += (len(hashes) - len(missing_hashes))

    print('\n--- Audit results ---')
    print(f'Local chunks: {total_chunks}')
    print(f'Chunks present in Qdrant: {present}')
    print(f'Chunks missing in Qdrant: {missing}')

    if missing == 0:
        print('All local chunks are present in Qdrant ✅')
    else:
        print('Files with missing chunks:')
        for fname, mh in missing_by_file.items():
            if mh:
                print(f' - {Path(fname).name}: {len(mh)} missing chunks')
        print('\nYou can re-upload missing chunks (dedup will skip existing ones).')
