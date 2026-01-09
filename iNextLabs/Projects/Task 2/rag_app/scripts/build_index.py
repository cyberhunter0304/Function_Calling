#!/usr/bin/env python3
"""Offline FAISS index builder.

Usage:
  - Set environment variable `OPENROUTER_API_KEY` before running.
  - Run this script from the project (rag_app) directory or directly with
    Python. It resolves paths relative to this script's parent directory.

What it does:
  - Loads chunks via `load_and_chunk_text` from `../data/inextlabs.txt`.
  - Generates embeddings with `embed_texts`.
  - Builds a FAISS IndexFlatL2 and adds embeddings.
  - Saves `embeddings/faiss_index.bin` and `embeddings/metadata.pkl`.
"""
from pathlib import Path
import sys
import pickle

import numpy as np
import faiss

# Make sure we can import the local `app` package by adding rag_app/ to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.embeddings import load_and_chunk_text, embed_texts


def main() -> None:
    data_file = ROOT / "data" / "inextlabs.txt"
    embeddings_dir = ROOT / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Load chunks
    print("Loading and chunking text...")
    chunks = load_and_chunk_text(str(data_file))
    if not chunks:
        print("No chunks produced from input file. Exiting.")
        return

    # Generate embeddings
    print(f"Generating embeddings for {len(chunks)} chunks...")
    emb_arr = embed_texts(chunks)
    if not isinstance(emb_arr, np.ndarray):
        raise RuntimeError("embed_texts must return a NumPy array")
    if emb_arr.dtype != np.float32:
        emb_arr = emb_arr.astype(np.float32)

    n, d = emb_arr.shape
    print(f"Embedding shape: {emb_arr.shape}")

    # Build FAISS index (L2)
    print("Building FAISS IndexFlatL2...")
    index = faiss.IndexFlatL2(d)
    index.add(emb_arr)

    # Save index and metadata
    index_file = embeddings_dir / "faiss_index.bin"
    metadata_file = embeddings_dir / "metadata.pkl"

    print(f"Saving FAISS index to {index_file}...")
    faiss.write_index(index, str(index_file))

    print(f"Saving metadata (chunks) to {metadata_file}...")
    with open(metadata_file, "wb") as f:
        pickle.dump(chunks, f)

    print("Index build complete.")


if __name__ == "__main__":
    main()
