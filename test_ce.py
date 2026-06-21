from sentence_transformers import CrossEncoder
import traceback

print("Starting...")

try:
    model = CrossEncoder("BAAI/bge-reranker-base")
    print("Loaded successfully!")
except Exception:
    traceback.print_exc()