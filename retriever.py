from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from ingest import COLLECTION_NAME, EMBEDDING_MODEL, VECTORSTORE_DIR

TOP_K = 4


def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )


def get_retriever(search_type: str = "similarity") -> VectorStoreRetriever:
    kwargs = {"k": TOP_K}
    if search_type == "mmr":
        kwargs.update(fetch_k=20, lambda_mult=0.5)  # 20 candidates -> 4 diverse
    return get_vectorstore().as_retriever(search_type=search_type, search_kwargs=kwargs)