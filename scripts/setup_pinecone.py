"""
setup_pinecone.py — one-time setup for the Pinecone vector index.

Run this script once after setting PINECONE_API_KEY in your .env file.
It will:
  1. Create the Pinecone serverless index (if it doesn't exist yet).
  2. Pre-populate the 'breeds' namespace with embeddings for all 37 Oxford Pets breeds.

Usage (from project root):
    set PYTHONPATH=%CD%           # Windows
    # export PYTHONPATH=$(pwd)    # Mac/Linux
    python scripts/setup_pinecone.py
"""

import logging
import sys
import os

# ---------------------------------------------------------------------------
# Bootstrap: load .env before importing anything that needs API keys.
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("setup_pinecone")


def check_env():
    """Verify required environment variables are set."""
    missing = []
    for key in ("PINECONE_API_KEY", "OPENAI_API_KEY"):
        if not os.getenv(key):
            missing.append(key)
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Set them in your .env file and retry.")
        sys.exit(1)
    logger.info("Environment variables OK.")


def main():
    check_env()

    # --- Step 1: trigger index creation via store._get_index() ---
    logger.info("Connecting to Pinecone and creating index if needed...")
    from agent.memory.store import _get_index, index_stats
    _get_index()
    logger.info("Index ready.")

    # --- Step 2: populate breed knowledge embeddings ---
    logger.info("Populating breed knowledge namespace (37 breeds)...")
    from agent.memory.manager import populate_breed_knowledge
    count = populate_breed_knowledge(force=False)

    if count == 0:
        logger.info("Breeds namespace already populated. Use force=True to re-embed.")
    else:
        logger.info(f"Successfully upserted {count} breed vectors.")

    # --- Step 3: print index stats ---
    stats = index_stats()
    logger.info(f"Index stats: {stats}")
    logger.info("Setup complete. You can now run the agent with vector memory enabled.")


if __name__ == "__main__":
    main()
