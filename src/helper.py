"""
Utility functions for document processing and embedding in the Medical Chatbot project.

This module provides modular, reusable helpers for:
- Loading PDF documents from a directory
- Splitting documents into text chunks
- Downloading HuggingFace embedding models

These functions are designed to be imported and used in scripts or notebooks that handle
vector database operations, embedding, and retrieval-augmented generation (RAG) pipelines.

Functions:
    load_pdf_file(data): Loads and returns PDF documents from a directory.
    text_split(extracted_data): Splits loaded documents into fixed-size overlapping chunks.
    download_hugging_face_embeddings(): Returns a HuggingFace embedding model instance.
"""

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf_file(data):
    """
    Loads all PDF files from the specified directory using LangChain's DirectoryLoader and PyPDFLoader.
    Args:
        data (str): Path to the directory containing PDF files.
    Returns:
        list: A list of loaded document objects.
    """
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """
    Splits documents into chunks for processing.
    Args:
        extracted_data (list): List of loaded document objects.
    Returns:
        list: List of text chunks with 500 characters per chunk and 20 characters overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_hugging_face_embeddings():
    """
    Downloads and initializes HuggingFace embeddings model.
    Returns:
        HuggingFaceEmbeddings: An instance of HuggingFace embeddings using the 'sentence-transformers/all-MiniLM-L6-v2' model.
        This model returns 384-dimensional embeddings.
    """
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings