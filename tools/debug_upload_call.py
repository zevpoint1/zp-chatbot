import os, types
import upload_to_qdrant as upl

def fake_post(url, headers=None, json=None, timeout=None):
    if url.endswith('/points/scroll'):
        class R:
            status_code = 200
            def json(self):
                return {'result': {'points': []}}
        return R()
    if url.endswith('/points/search'):
        class R:
            status_code = 200
            def json(self):
                return {'result': [{'id':1}]}
        return R()
    class R:
        status_code = 404
        def json(self):
            return {}
    return R()


def fake_put(url, headers=None, data=None, timeout=None):
    class R:
        status_code = 200
        def json(self):
            return {}
    return R()

upl.requests = types.SimpleNamespace(post=fake_post, put=fake_put)
upl.embed_batch = lambda texts: [[0.1]*8]

os.environ['QDRANT_VALIDATE_UPLOAD'] = '1'
print('call upload_texts ->', upl.upload_texts(['this is a test'], metadata={'source':'unit_test'}))
