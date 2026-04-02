import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

def load_document(file_path: str):
    """
    Currently takes in a pdf file (locally) and loads it in using PyPDF
    
    :params file_path: The path of where the document to be loaded is.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    
    return PyPDFLoader(file_path=file_path).load()

def split_document_into_chunks(document, chunk_size=1000, chunk_overlap=200):
    """
    Takes a document object and splits it into chunks accordingly
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(document)

def load_embeddings_model_from_HF(model_name=None):
    """
    Load embedding model from HuggingFace.

    :param model_name: name of the embedding model to use.
    """

    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(model_name=model_name)

def document_already_exists(vector_store: PineconeVectorStore, source: str) -> bool:
    """
    Checks if the document/source to be added is already in the vector store.

    :param vector_store: Vector store the document will be added to
    :param source: identify of the document, probs just the file name (simplicity)
    """
    results = vector_store.similarity_search(
        query="check if document exist",
        k = 1,
        filter= {'source': f'{source}'}
    )
    return len(results) > 0

def store_embeddings(splits, embeddings):
    """
    Stores embedded chunks into the vector store

    :param splits: a list of chunks representing the document
    :param embeddings: Embedding's model
    """
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("rag-index")
    vector_store = PineconeVectorStore(embedding=embeddings, index=index)

    source = splits[0].metadata.get("source", "")

    # handle already existing documents.
    if document_already_exists(vector_store, source=source):
        print(f"{source} already in the vector store, skipping")
    else:
        print(f"Adding {source} to the vector store")
        vector_store.add_documents(documents=splits)
        print(f"Added {len(splits)} chunks to the vector store")
    return vector_store