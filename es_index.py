# es_index.py
# Elasticsearch index setup utilities.

from elasticsearch import Elasticsearch
from config import ES_ENDPOINT, ES_INDEX, EMBED_DIMS, ES_USER, ES_PASS

def get_es_client():
    """Return an Elasticsearch client using basic auth from config."""
    return Elasticsearch(
        ES_ENDPOINT, 
        verify_certs=False,
        basic_auth=(ES_USER, ES_PASS)
    )

def test_connection(es_client):
    """Test Elasticsearch connection with better error handling."""
    try:
        # Try to get cluster info instead of ping
        info = es_client.info()
        print(f"Connected to Elasticsearch cluster: {info.get('cluster_name', 'unknown')}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False

def create_index_if_not_exists(es_client, index_name=None):
    """
    Create the specified index if it doesn't exist.
    If no index_name is provided, uses ES_INDEX from config.
    """
    if index_name is None:
        index_name = ES_INDEX
    
    try:
        # Check if index exists
        if es_client.indices.exists(index=index_name):
            print(f"Index '{index_name}' already exists.")
            return True
        
        # Create index with mapping
        print(f"Creating index '{index_name}'...")
        mapping = {
            "mappings": {
                "properties": {
                    "benchmark": {"type": "keyword"},
                    "subset": {"type": "keyword"},
                    "split_hint": {"type": "keyword"},
                    "id_in_benchmark": {"type": "keyword"},
                    "text": {"type": "text"},
                    "answer": {"type": "text"},
                    "topic": {"type": "keyword"},
                    "topic_embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "difficulty": {"type": "integer"},
                    "relevance_hint": {"type": "float"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBED_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        
        es_client.indices.create(index=index_name, body=mapping)
        print(f"Successfully created index '{index_name}'.")
        return True
        
    except Exception as e:
        print(f"Error creating index '{index_name}': {e}")
        return False

def ensure_index(es_client):
    """
    Create index with a dense_vector field for embeddings if it does not exist.
    Compatible with Elasticsearch 7.x+ without requiring additional plugins.
    """
    try:
        # Test connection first
        if not test_connection(es_client):
            print(f"Cannot connect to Elasticsearch at {ES_ENDPOINT}")
            print("Please ensure Elasticsearch is running and accessible")
            return False
        
        # Create index if it doesn't exist
        return create_index_if_not_exists(es_client)
        
    except Exception as e:
        print(f"Error in ensure_index: {e}")
        print(f"Please ensure Elasticsearch is running at {ES_ENDPOINT}")
        return False
