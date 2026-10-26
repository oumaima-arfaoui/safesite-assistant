import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"
COLLECTION_NAME = "safesite"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800    
CHUNK_OVERLAP = 120   
MIN_CHUNK_CHARS = 60  


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        for page_doc in PyMuPDFLoader(str(pdf)).load():
            page_doc.metadata = {
                "source": pdf.name,
                "page": page_doc.metadata.get("page", 0) + 1,  # 1-based
            }
            docs.append(page_doc)
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    return [c for c in chunks if len(c.page_content.strip()) >= MIN_CHUNK_CHARS]


def chunk_id(chunk: Document) -> str:
    key = f"{chunk.metadata['source']}|{chunk.metadata['page']}|{chunk.page_content}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def main() -> None:
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} pages from {len({d.metadata['source'] for d in docs})} files.")

    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # Deterministic IDs -> re-running never duplicates
    by_id = {chunk_id(c): c for c in chunks}

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
    )

    existing = set(store.get(ids=list(by_id))["ids"])
    new_ids = [i for i in by_id if i not in existing]
    if new_ids:
        store.add_documents([by_id[i] for i in new_ids], ids=new_ids)

    print(f"Added {len(new_ids)} new chunks. Store now holds {store._collection.count()}.")


if __name__ == "__main__":
    main()