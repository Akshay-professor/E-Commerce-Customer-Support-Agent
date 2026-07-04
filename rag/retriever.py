import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

BASE_DOC_PATH = "rag/docs"
BASE_DB_PATH = "rag/vectorstores"

# Google API embeddings — no local torch/transformers/sentence-transformers, so
# the app stays lightweight and deploys cleanly on Streamlit Cloud. Uses the same
# Gemini key as the chat model.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

def build_vector_store(category: str):
    """
    Builds a specific vector store for a given category (e.g., 'returns', 'shipping').
    """
    source_path = os.path.join(BASE_DOC_PATH, category)
    db_path = os.path.join(BASE_DB_PATH, category)

    if not os.path.exists(source_path):
        print(f"[rag] Folder {source_path} not found. Skipping.")
        return None

    # Load PDFs from the specific category folder
    loader = DirectoryLoader(source_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()

    if not docs:
        print(f"[rag] No documents found in {category}.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Create and Save local FAISS index
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(db_path)
    print(f"[rag] Built index for '{category}' at {db_path}")
    return db

def get_retriever(category: str):
    """
    Loads the specific vector store for the requested category.
    """
    db_path = os.path.join(BASE_DB_PATH, category)
    
    # Check if DB exists; if not, try to build it
    if not os.path.exists(db_path):
        db = build_vector_store(category)
        if not db:
            return None
    else:
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    
    return db.as_retriever(search_kwargs={'k': 3})


# Cache loaded vector stores so scored search doesn't reload FAISS each call.
_stores: dict = {}


def _get_store(category: str):
    if category in _stores:
        return _stores[category]
    db_path = os.path.join(BASE_DB_PATH, category)
    if not os.path.exists(db_path):
        db = build_vector_store(category)
    else:
        db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)
    _stores[category] = db
    return db


def search_with_scores(category: str, query: str, k: int = 3):
    """
    Semantic search that also returns a normalized retrieval-quality score in
    [0, 1] (1 = best) for the confidence calculation, alongside the docs.

    FAISS returns L2 distance (lower = closer); we map the best distance to a
    0..1 quality via 1 / (1 + distance). Returns (docs, best_quality).
    """
    db = _get_store(category)
    if db is None:
        return [], 0.0

    results = db.similarity_search_with_score(query, k=k)
    if not results:
        return [], 0.0

    docs = [doc for doc, _ in results]
    best_distance = min(score for _, score in results)
    best_quality = 1.0 / (1.0 + float(best_distance))
    return docs, best_quality


# Initialize all DBs when running this script directly
if __name__ == "__main__":
    for cat in ["returns", "shipping", "general","cancel"]:
        build_vector_store(cat)