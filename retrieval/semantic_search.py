from langchain_community.vectorstores import FAISS
from ingestion.vector_store import TfidfEmbeddings
import pickle
import os

VECTOR_DB_PATH = "vector_store/dealsense_faiss"

def load_vector_store():
    # 🔑 Load the SAME fitted TF-IDF vectorizer
    tfidf_path = os.path.join(VECTOR_DB_PATH, "tfidf.pkl")
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)

    embeddings = TfidfEmbeddings(vectorizer)

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

def semantic_search(query, k=3):
    vector_db = load_vector_store()
    # IMPORTANT: return scores
    return vector_db.similarity_search_with_score(query, k=k)
