import os
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


def load_embeddings_model_from_HF(model_name=None):
    """
    Load embedding model from HuggingFace.

    :param model_name: name of the embedding model to use.
    """

    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    return HuggingFaceEmbeddings(model_name=model_name)


def get_vector_store(embedding_model):
    api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=api_key)
    index = pc.Index("rag-index")
    return PineconeVectorStore(embedding=embedding_model, index=index)