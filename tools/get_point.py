import os
import requests
import json

Q=os.getenv('QDRANT_URL')
K=os.getenv('QDRANT_API_KEY')
C=os.getenv('QDRANT_COLLECTION','ev_kb')
ID=os.getenv('QDRANT_POINT_ID')
if ID is None:
    print('Set QDRANT_POINT_ID env var to the id to fetch')
    raise SystemExit(2)
resp=requests.get(f"{Q}/collections/{C}/points/{ID}", headers={'api-key':K}, timeout=10)
print('status',resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2))
except Exception as e:
    print('Failed to parse JSON', e)
    print(resp.text)
