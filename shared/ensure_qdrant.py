"""
ensure_qdrant.py
Simple script to check and fix Qdrant collection
"""
import os
import requests
import json

# Prefer the configured embedding dimension so collection matches model
from shared.embed_text import EMBEDDING_DIMENSION

# Get config from environment
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "ev_kb")

def check_and_fix():
    """Check if collection exists, create if missing"""
    
    # Check if collection exists
    print(f"Checking collection: {COLLECTION}")
    response = requests.get(
        f"{QDRANT_URL}/collections/{COLLECTION}",
        headers={"api-key": QDRANT_API_KEY}
    )
    
    if response.status_code == 404:
        # Collection missing - create it
        print("❌ Collection missing - creating...")
        
        response = requests.put(
            f"{QDRANT_URL}/collections/{COLLECTION}",
            headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
            json={
                "vectors": {
                    "size": EMBEDDING_DIMENSION,  # determined from EMBEDDING_MODEL
                    "distance": "Cosine"
                }
            }
        )
        print(f"Created collection with vector size: {EMBEDDING_DIMENSION}")
        
        if response.status_code in [200, 201]:
            print("✅ Collection created!")
            print("⚠️  NOW YOU NEED TO ADD YOUR DATA")
            print("   Run your data ingestion script")
            return False  # No data yet
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
    
    elif response.status_code == 200:
        # Collection exists - check if it has data
        info = response.json()['result']
        points = info['points_count']
        
        if points == 0:
            print(f"⚠️  Collection exists but EMPTY (0 documents)")
            print("   You need to run your data ingestion script!")
            return False
        else:
            print(f"✅ Collection OK - {points} documents ready")
            return True
    
    else:
        print(f"❌ Error: {response.status_code}")
        return False

if __name__ == "__main__":
    check_and_fix()