import os
from text_chunker import load_and_chunk_documents
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

SCRAPED_DIR = "scraped_content"

# Load and chunk documents
chunks = load_and_chunk_documents(SCRAPED_DIR, chunk_size=1000, chunk_overlap=200)

# Embed chunks using an open-source model
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in FAISS vector database
faiss_db = FAISS.from_documents(chunks, embedder)
faiss_db.save_local("faiss_index")

print("Embedding and storage complete. FAISS index saved to 'faiss_index'.")
