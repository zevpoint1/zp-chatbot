import os
import sys
from pathlib import Path
from typing import List

# ensure imports work when running the script
sys.path.insert(0, os.getcwd())

from shared.chunking import chunk_with_metadata
from upload_to_qdrant import fetch_existing_hashes, text_hash, upload_texts

UPLOADS_DIR = Path(os.getenv('UPLOADS_DIR', 'uploads'))
BATCH_SIZE = int(os.getenv('REINGEST_BATCH_SIZE', '16'))


def chunks_for_file(path: Path):
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
            for sc in sub_chunks:
                sc['metadata']['line_index'] = i
            chunks.extend(sub_chunks)
    else:
        text = path.read_text(encoding='utf-8', errors='ignore')
        chunks = chunk_with_metadata(text, source_file=path.name, file_type=path.suffix.lstrip('.'))

    for c in chunks:
        c['chunk_hash'] = text_hash(c['text'])
    return chunks


if __name__ == '__main__':
    print('Reingest missing: scanning uploads directory', UPLOADS_DIR)

    files = [f for f in UPLOADS_DIR.glob('**/*') if f.is_file()]
    total_to_upload = 0
    total_uploaded = 0

    for f in files:
        chunks = chunks_for_file(f)
        if not chunks:
            continue

        # compute which chunks are missing by comparing hashes to current Qdrant index
        existing = fetch_existing_hashes()
        missing = [c for c in chunks if c['chunk_hash'] not in existing]

        if not missing:
            print(f"{f.name}: all chunks already present, skipping")
            continue

        print(f"{f.name}: {len(missing)} missing chunks; re-uploading in batches of {BATCH_SIZE}...")
        total_to_upload += len(missing)

        # upload in batches
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i:i+BATCH_SIZE]
            texts = [c['text'] for c in batch]
            metadata = {'source_file': f.name}
            ok = upload_texts(texts, metadata=metadata)
            if ok:
                uploaded_count = len(texts)
                total_uploaded += uploaded_count
                print(f'  Uploaded batch {i//BATCH_SIZE + 1}: {uploaded_count} texts')
            else:
                print(f'  Failed to upload batch {i//BATCH_SIZE + 1}; stopping for file {f.name}')
                break

    print('\n--- Reingest summary ---')
    print(f'Total missing chunks identified: {total_to_upload}')
    print(f'Total chunks successfully uploaded: {total_uploaded}')
    if total_to_upload == total_uploaded:
        print('Reingest complete ✅')
    else:
        print('Reingest finished with some failures. Check logs for details.')
