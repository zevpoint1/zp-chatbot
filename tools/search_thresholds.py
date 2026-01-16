import os, requests, json, sys
sys.path.insert(0, os.getcwd())
from shared.embed_text import embed
Q=os.getenv('QDRANT_URL')
K=os.getenv('QDRANT_API_KEY')
C=os.getenv('QDRANT_COLLECTION','ev_kb')
q='test vector'
vec=embed(q)
for thr in [0.3, 0.4, 0.45, 0.5, 0.6, 0.7]:
    payload={'vector': vec, 'limit': 5, 'with_payload': True, 'with_vector': False, 'score_threshold': thr}
    resp=requests.post(f"{Q}/collections/{C}/points/search", headers={'api-key':K,'Content-Type':'application/json'}, json=payload, timeout=10)
    print(f"thr={thr} status={resp.status_code} len(result)={len(resp.json().get('result',[]))}")
    if resp.status_code==200 and resp.json().get('result'):
        print('top score', resp.json()['result'][0]['score'])
