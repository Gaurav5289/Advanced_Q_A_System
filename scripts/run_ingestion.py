import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion.pipeline import run_ingestion_pipeline

if __name__ == "__main__":
    run_ingestion_pipeline()