import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

SCRAPED_DIR = "scraped_content"

def load_and_chunk_documents(scraped_dir, chunk_size=1000, chunk_overlap=200):
    """
    Loads all .txt files from scraped_dir and chunks them using RecursiveCharacterTextSplitter.
    Returns a list of Document chunks.
    """
    documents = []
    for filename in os.listdir(scraped_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(scraped_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()
                documents.append(Document(page_content=text, metadata={"source": filename}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in documents:
        chunks.extend(splitter.split_documents([doc]))
    return chunks

if __name__ == "__main__":
    chunks = load_and_chunk_documents(SCRAPED_DIR, chunk_size=1000, chunk_overlap=200)
    print(f"Total chunks: {len(chunks)}")
    if chunks:
        print("First chunk:", chunks[0].page_content)
