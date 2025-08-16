import sys
import os
from core.config import settings

print("🔑 Google API Key:", settings.GOOGLE_API_KEY[:5], "...")
print("🔑 Llama API Key:", settings.LLAMA_CLOUD_API_KEY[:5], "...")
print("🔑 Pinecone API Key:", settings.PINECONE_API_KEY[:5], "...")
print("📦 Pinecone Index:", settings.PINECONE_INDEX_NAME)
print("⚡ Vector Store Provider:", settings.VECTOR_STORE_PROVIDER)



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.pipeline import run_ingestion_pipeline

if __name__ == "__main__":
    run_ingestion_pipeline()