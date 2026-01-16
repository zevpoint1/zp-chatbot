import os
import sys
import json
from typing import List

# ensure project path
sys.path.insert(0, os.getcwd())

from shared.embed_text import embed
from shared.query_pipeline import Config
import requests

Q = Config.QDRANT_URL
K = Config.QDRANT_API_KEY
C = Config.COLLECTION

# Representative queries (short, mid, and long forms)
QUERIES = [
    "charging speed for electric cars",
    "safety recommendations for EV charging",
    "ZEVPOINT AVEO X1 specifications",
    "how long to charge at 7.5kW",
    "policy on EV charger maintenance",
]


def search_with_vector(vec, limit=3, score_threshold=None):
    payload = {"vector": vec, "limit": limit, "with_payload": True, "with_vector": False}
    if score_threshold is not None:
        payload["score_threshold"] = score_threshold
    resp = requests.post(f"{Q}/collections/{C}/points/search", headers={'api-key':K,'Content-Type':'application/json'}, json=payload, timeout=20)
    return resp.status_code, resp.json()


if __name__ == '__main__':
    print(f"Using Qdrant: {Q} collection: {C}")
    print("Configured MIN_SIMILARITY_SCORE: not set in Config (disabled)\n")

    for q in QUERIES:
        vec = embed(q)
        print('---')
        print(f"Query: {q}")

        # Search with configured threshold (disabled by default)
        st = None
        code, data = search_with_vector(vec, score_threshold=st)
        print(f"With score_threshold={'disabled'} -> status={code} hits={len(data.get('result', []))}")
        if data.get('result'):
            for h in data['result']:
                print(f"  id={h.get('id')} score={h.get('score')} payload_keys={list(h.get('payload', {}).keys())}")

        # Search without threshold
        code2, data2 = search_with_vector(vec, score_threshold=None)
        print(f"Without score_threshold -> status={code2} hits={len(data2.get('result', []))}")
        if data2.get('result'):
            top = data2['result'][0]
            print(f"  top score={top.get('score')} id={top.get('id')} payload_sample={top.get('payload', {}).get('text', '')[:80].replace('\n',' ')}")

    print('\nEvaluation completed')