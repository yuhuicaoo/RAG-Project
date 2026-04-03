from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import time
from pathlib import Path
from ingest import document_already_exists, ingest_document
from query import get_agent
from utils import load_embeddings_model_from_HF, get_vector_store


@st.cache_resource
def setup():
    embedding_model = load_embeddings_model_from_HF()
    vector_store = get_vector_store(embedding_model)
    agent = get_agent(vector_store)
    return embedding_model, vector_store, agent
   
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
    embedding_model, vector_store, agent = setup()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
    # file uploader
    uploaded_files = st.file_uploader(
        "Upload files", accept_multiple_files=True, type="pdf"
    )

    if uploaded_files:
        # only keep documents that arent in the vector store
        new_docs = [
            file for file in uploaded_files 
            if not document_already_exists(vector_store, file.name)]
        
        # upload the new documents to the vector store
        for uploaded_file in new_docs:
            ingest_document(uploaded_file, embedding_model)


    st.markdown("""
        <style>
        [data-testid="stMarkdownContainer"] p { font-size: 20px; }
        </style>""", 
        unsafe_allow_html=True
    )

    # display the uploaded documents on the sidebar
    for file in uploaded_files:
        st.markdown(f":material/picture_as_pdf: {Path(file.name).stem}")
        time.sleep(0.05)
        

if query := st.chat_input("Hello, how can I help?"):
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    # get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.write_stream(response_generator(agent, query))

    st.session_state.messages.append({"role": "assistant", "content": response})