import os
import pickle
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA_PATH = "./data/linux commands.pdf"
INDEX_DIR = "./indices"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 75):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def build_indices():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Source PDF file not found at {DATA_PATH}")

    print("Extracting text from PDF...")
    reader = PdfReader(DATA_PATH)
    full_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
    chunks = chunk_text(full_text)

    print(f"Total chunks created: {len(chunks)}")
    print("Building FAISS Dense Index...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)

    print("Building BM25 Sparse Index...")
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(faiss_index, f"{INDEX_DIR}/faiss.index")
    
    with open(f"{INDEX_DIR}/bm25.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    with open(f"{INDEX_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
        
    print("Indices built and saved successfully!")

if __name__ == "__main__":
    build_indices()