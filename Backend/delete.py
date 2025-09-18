from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://71cc6272-375c-4d4c-bdf6-d395f077e063.europe-west3-0.gcp.cloud.qdrant.io:6333",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.rhjtxRG7yF6vHVxb9ZzuphzSa60hhlc7x_goMhY0iII"
)

client.delete_collection(collection_name="audio_kb")
