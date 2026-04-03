import logging
import os
from typing import Any, List
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import Blob, PyMuPDFLoader
from langchain_community.document_loaders.parsers.pdf import (
    PyMuPDFParser,
)
from langchain_core.documents import Document
from io import BytesIO

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

class BytesIOPyMuPDFLoader(PyMuPDFLoader):
    """
    Load `PDF` files using `PyMuPDF` from a BytesIO stream.
    
    Needed a way to parse through streamlit uploaded files into the LangChain PDF loader.
    Solution found from --> https://github.com/langchain-ai/langchain/issues/6265#issuecomment-1929210954
    """

    def __init__(
        self,
        pdf_stream: BytesIO,
        *,
        extract_images: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with a BytesIO stream."""
        try:
            import fitz  # noqa:F401
        except ImportError:
            raise ImportError(
                "`PyMuPDF` package not found, please install it with "
                "`pip install pymupdf`"
            )
        # We don't call the super().__init__ here because we don't have a file_path.
        self.pdf_stream = pdf_stream
        self.extract_images = extract_images
        self.text_kwargs = kwargs

    def load(self, **kwargs: Any) -> List[Document]:
        """Load file."""
        if kwargs:
            logging.warning(
                f"Received runtime arguments {kwargs}. Passing runtime args to `load`"
                f" is deprecated. Please pass arguments during initialization instead."
            )

        text_kwargs = {**self.text_kwargs, **kwargs}

        # Use 'stream' as a placeholder for file_path since we're working with a stream.
        blob = Blob.from_data(self.pdf_stream.getvalue(), path="stream")

        parser = PyMuPDFParser(
            text_kwargs=text_kwargs, extract_images=self.extract_images
        )

        return parser.parse(blob)