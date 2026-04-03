from dotenv import load_dotenv
load_dotenv()


import streamlit as st
import time
import os
from ingest import document_already_exists, ingest_document
from query import get_agent
from utils import load_embeddings_model_from_HF, get_vector_store


DOCUMENTS_DIR = "documents"

@st.cache_resource
def setup():
    embedding_model = load_embeddings_model_from_HF()
    vector_store = get_vector_store(embedding_model)

    new_docs = [
        file
        for file in os.listdir(DOCUMENTS_DIR)
        if file.endswith(".pdf")
        and not document_already_exists(vector_store, os.path.join(DOCUMENTS_DIR, file))
    ]

    if new_docs:
        # ingest all documents in data folder
        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                ingest_document(file_path, embedding_model)

    agent = get_agent(vector_store)
    return agent
   
def response_generator(agent, query):
    events = list(agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values"
            ))
    
    response = events[-1]["messages"][-1].content
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("LLM + RAG Assistant")

with st.spinner("Loading model and documents..."):
    agent = setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Hello, how can I help?"):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.write_stream(response_generator(agent, query))

    st.session_state.messages.append({"role": "assistant", "content": response})