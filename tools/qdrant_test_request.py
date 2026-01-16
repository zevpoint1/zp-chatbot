import os, requests, json, sys
sys.path.insert(0, os.getcwd())
from shared.embed_text import embed

Q=os.getenv('QDRANT_URL')
K=os.getenv('QDRANT_API_KEY')
C=os.getenv('QDRANT_COLLECTION','ev_kb')
q='test vector'
vec=embed(q)
print('=== Searching with score_threshold=0.7 ===')
payload={'vector': vec, 'limit': 10, 'with_payload': True, 'with_vector': True, 'score_threshold': 0.7}
resp=requests.post(f"{Q}/collections/{C}/points/search", headers={'api-key':K,'Content-Type':'application/json'}, json=payload, timeout=10)
print('status', resp.status_code)
print('response', resp.text)

print('\n=== Searching WITHOUT score_threshold ===')
payload={'vector': vec, 'limit': 10, 'with_payload': True, 'with_vector': True}
resp=requests.post(f"{Q}/collections/{C}/points/search", headers={'api-key':K,'Content-Type':'application/json'}, json=payload, timeout=10)
print('status', resp.status_code)
print('response', resp.text)
print('Payload keys:', list(payload.keys()))
resp=requests.post(f"{Q}/collections/{C}/points/search", headers={'api-key':K,'Content-Type':'application/json'}, json=payload, timeout=10)
print('status', resp.status_code)
print('response', resp.text)
