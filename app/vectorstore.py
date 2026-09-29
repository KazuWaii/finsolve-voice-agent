from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

PERSIST_DIR = Path(__file__).resolve().parents[1] / "resources" / "vectorstore"
COLLECTION_NAME = "faq"

_embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collection():
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)


def index_chunks(chunks):
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=_embedding_fn)
    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[c.metadata for c in chunks],
    )


def query(question, n_results=3):
    collection = get_collection()
    result = collection.query(query_texts=[question], n_results=n_results)
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]