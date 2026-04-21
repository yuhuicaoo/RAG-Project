# config.py
import os
import streamlit as st
from dotenv import load_dotenv

def load_secrets():
    try:
        # Streamlit Cloud - nested secrets
        nested_secrets = {
            "groq": ["GROQ_API_KEY"],
            "pinecone": ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"],
            "langsmith": ["LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_ENDPOINT", "LANGSMITH_PROJECT"],
        }
        for section, keys in nested_secrets.items():
            for key in keys:
                os.environ[key] = st.secrets[section][key]
    except Exception:
        # Local dev fallback
        load_dotenv()