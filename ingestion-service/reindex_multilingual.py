import logging
import os
import shutil
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings
from app.embeddings import generate_embeddings_batch, get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("reindex_migration")

def reindex_all_documents():
    """
    Migration script to re-embed all documents in ChromaDB with the new
    multilingual embedding model (intfloat/multilingual-e5-base, 768 dim).
    """
    logger.info(f"Starting vector DB migration to model: {settings.EMBEDDING_MODEL_NAME} (768-dim)...")
    
    # 1. Connect to existing ChromaDB
    persist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_db"))
    if not os.path.exists(persist_dir):
        logger.warning(f"No existing Chroma directory found at {persist_dir}. Creating fresh.")
        os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # 2. Get existing collection
    try:
        old_col = client.get_collection(name=settings.COLLECTION_NAME)
        data = old_col.get(include=["documents", "metadatas"])
        ids = data.get("ids", [])
        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        logger.info(f"Found {len(ids)} existing document chunks to re-embed.")
    except Exception as e:
        logger.info(f"No previous collection found or collection empty: {e}")
        ids, documents, metadatas = [], [], []

    # 3. If there are documents, compute new 768-dim embeddings
    if ids and documents:
        logger.info(f"Re-embedding {len(documents)} chunks with {settings.EMBEDDING_MODEL_NAME}...")
        new_embeddings = generate_embeddings_batch(documents, is_query=False)
        logger.info(f"Generated {len(new_embeddings)} new embeddings of dim={len(new_embeddings[0])}.")

        # 4. Re-create collection with cosine distance
        client.delete_collection(name=settings.COLLECTION_NAME)
        new_col = client.create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # 5. Upsert newly embedded chunks
        new_col.upsert(
            ids=ids,
            documents=documents,
            embeddings=new_embeddings,
            metadatas=metadatas,
        )
        logger.info(f"Successfully migrated {len(ids)} chunks to {settings.COLLECTION_NAME} with multilingual embeddings!")
    else:
        # Create empty collection if none existed
        try:
            client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("Created clean collection with 768-dim space.")
        except Exception as ex:
            logger.warning(f"Error creating collection: {ex}")

    # 6. Reset Semantic Cache DB in rag-chat-service (old 384-dim cache entries are incompatible)
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rag-chat-service", "data", "semantic_cache_db"))
    if os.path.exists(cache_dir):
        try:
            shutil.rmtree(cache_dir)
            logger.info(f"Cleared old 384-dim semantic cache at {cache_dir}")
        except Exception as ex:
            logger.warning(f"Could not clear cache dir: {ex}")

    logger.info("Multilingual Vector DB Migration completed successfully!")

if __name__ == "__main__":
    reindex_all_documents()
