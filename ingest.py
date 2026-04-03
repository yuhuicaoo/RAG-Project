import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from utils import get_vector_store, BytesIOPyMuPDFLoader
import io

load_dotenv()

def load_document(uploaded_file):
    """
    Currently takes in a pdf file (locally) and loads it in using PyPDF
    
    :params file_path: The path of where the document to be loaded is.
    """
    uploaded_file.seek(0)
    loader = BytesIOPyMuPDFLoader(io.BytesIO(uploaded_file.read()))
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = uploaded_file.name
    
    return docs

def split_document_into_chunks(document, chunk_size=1000, chunk_overlap=200):
    """
    Takes a document object and splits it into chunks accordingly
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(document)

def document_already_exists(vector_store: PineconeVectorStore, source: str) -> bool:
    """
    Checks if the document/source to be added is already in the vector store.

    :param vector_store: Vector store the document will be added to
    :param source: identify of the document, probs just the file name (simplicity)
    """
    results = vector_store.similarity_search(
        query="check if document exist",
        k = 1,
        filter= {"source": {"$eq": source}}
    )
    return len(results) > 0

def store_embeddings(splits, embedding_model):
    """
    Stores embedded chunks into the vector store

    :param splits: a list of chunks representing the document
    :param embeddings: Embedding's model
    """
    vector_store = get_vector_store(embedding_model)

    source = splits[0].metadata.get("source", "")

    # handle already existing documents.
    if document_already_exists(vector_store, source=source):
        print(f"{source} already in the vector store, skipping")
    else:
        print(f"Adding {source} to the vector store")
        vector_store.add_documents(documents=splits)
        print(f"Added {len(splits)} chunks to the vector store")
    return vector_store

def ingest_document(uploaded_file, embeddings):
    return store_embeddings(split_document_into_chunks(load_document(uploaded_file)), embedding_model=embeddings)
