import sys, os
sys.path.insert(0, os.getcwd())
from upload_to_qdrant import upload_texts

print(upload_texts(["This is a small test to upload to Qdrant"], metadata={"source":"unit_test"}))
