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
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec, Index
    from langchain_pinecone import PineconeVectorStore
except ImportError as e:
    print("Error: Required packages not found. Please install them using:")
    print("pip install -r requirements.txt")
    sys.exit(1)

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

        # Create Pinecone index if it doesn't exist
        print_step(f"Checking if index '{index_name}' exists...")
        if index_name not in [idx.name for idx in pc.list_indexes()]:
            print_step(f"Creating new index: {index_name}")
            pc.create_index(
                name=index_name,
                dimension=384,  # for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            print_step(f"Index '{index_name}' created successfully", "DONE")
        else:
            print_step(f"Using existing index: {index_name}", "SKIP")

        # Upsert documents using LangChain's PineconeVectorStore
        print_step(f"Upserting {len(text_chunks)} chunks to Pinecone...")
        docsearch = PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=index_name,
            embedding=embeddings,
        )
        print_step("Upsert completed successfully", "DONE")
        print("\n✨ Indexing process completed successfully! ✨")

    except Exception as e:
        print_step(f"An error occurred: {str(e)}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
