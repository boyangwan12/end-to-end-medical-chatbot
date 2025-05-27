"""
Script for creating and populating a Pinecone vector index for the Medical Chatbot project.

Only run this script when you are updating the PDF documents in the data/ directory.

This script performs the following steps:
1. Loads PDF documents from the data/ directory
2. Splits documents into text chunks
3. Initializes HuggingFace embeddings
4. Connects to Pinecone
5. Creates a new index if it doesn't exist
6. Upserts document vectors to the index
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from tqdm import tqdm

# Import helper functions
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings

# Import Pinecone and LangChain components
from pinecone import Pinecone, ServerlessSpec, Index

def print_step(step: str, status: str = "START") -> None:
    """Print a formatted step message with status."""
    status_map = {
        "START": "🔄",
        "DONE": "✅",
        "SKIP": "⏩",
        "ERROR": "❌"
    }
    print(f"\n{status_map.get(status, '➡️')} {step}")

def main():
    """Main function to execute the indexing pipeline."""
    try:
        # Load environment variables
        print_step("Loading environment variables...")
        load_dotenv()
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        print_step("Environment variables loaded", "DONE")

        # Load and process PDF documents
        print_step("Loading PDF documents from data/ directory...")
        extracted_data = load_pdf_file(data='data/')
        print_step(f"Loaded {len(extracted_data)} documents", "DONE")

        print_step("Splitting documents into chunks...")
        text_chunks = text_split(extracted_data)
        print_step(f"Split into {len(text_chunks)} chunks", "DONE")

        # Initialize embeddings
        print_step("Initializing HuggingFace embeddings...")
        embeddings = download_hugging_face_embeddings()
        print_step("Embeddings initialized", "DONE")

        # Initialize Pinecone client
        print_step("Initializing Pinecone client...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "medicalbot"
        print_step("Pinecone client initialized", "DONE")

        # Robust index creation and readiness check
        print_step(f"Checking if index '{index_name}' exists...")
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            print_step(f"Index '{index_name}' does not exist. Creating it...")
            pc.create_index(
                name=index_name,
                dimension=384,  # for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            # Wait for index to be ready
            import time
            while True:
                status = pc.describe_index(index_name).status['ready']
                if status:
                    print_step(f"Index '{index_name}' is ready.", "DONE")
                    break
                print_step("Waiting for index to be ready...", "SKIP")
                time.sleep(2)
        else:
            print_step(f"Index '{index_name}' already exists.", "SKIP")
        index = pc.Index(index_name)

        # Upsert documents directly using Pinecone v7+ API (optimized batching)
        print_step(f"Upserting {len(text_chunks)} chunks to Pinecone in batches...")
        batch_size = 32  # Tune this based on your memory/cpu
        total_chunks = len(text_chunks)
        for start in range(0, total_chunks, batch_size):
            end = min(start + batch_size, total_chunks)
            batch_chunks = text_chunks[start:end]
            texts = [chunk.page_content for chunk in batch_chunks]
            # Use batch embedding for speed (if available)
            try:
                vectors_batch = embeddings.embed_documents(texts)
            except Exception:
                # Fallback to single embedding if batch not supported
                vectors_batch = [embeddings.embed_query(text) for text in texts]
            batch_vectors = []
            for idx, chunk in enumerate(batch_chunks):
                metadata = getattr(chunk, 'metadata', {}).copy() if hasattr(chunk, 'metadata') else {}
                metadata["text"] = chunk.page_content
                batch_vectors.append({
                    "id": f"chunk-{start + idx}",
                    "values": vectors_batch[idx],
                    "metadata": metadata
                })
            try:
                response = index.upsert(vectors=batch_vectors)
                print(f"✅ Upserted chunks {start}-{end-1} (response: {response})")
            except Exception as upsert_error:
                print(f"❌ Error upserting batch {start}-{end-1}: {upsert_error}")
                raise
        print_step("Upsert completed successfully", "DONE")
        print("\n Indexing process completed successfully! ")

    except Exception as e:
        print_step(f"An error occurred: {str(e)}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
