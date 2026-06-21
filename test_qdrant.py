import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    port=None,
    api_key=os.getenv("QDRANT_API_KEY"),
    prefer_grpc=False,
    trust_env=False,
)

print(client.get_collections())
