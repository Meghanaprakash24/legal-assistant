import os
from dotenv import load_dotenv

load_dotenv()

print("QDRANT_URL:", os.getenv("QDRANT_URL"))
print("API KEY EXISTS:", bool(os.getenv("QDRANT_API_KEY")))
print("COLLECTION:", os.getenv("COLLECTION_NAME"))