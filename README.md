# 🔍 RAG-Project
## Table of Contents
- [Purpose](#purpose)
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Next Steps](#-next-steps)
- [Contact](#-contact)

## 🎯 Purpose
As a student constantly navigating multiple dense (and often very technical) research papers, I started this project to make it easier to ask questions directly against a paper and get accurate, grounded answers, rather than skimming endlessly or losing context across long documents.

## 🧠 Overview
<p align="justify">
This project implements an end-to-end Retrieval Augmented Generation (RAG) pipeline that enables intelligent question-answering over custom document corpora. Documents are embedded using a HuggingFace embedding model, stored and retrieved from a Pinecone vector database, and responses are generated using Llama4 Scout (via Groq). Built on the LangChain framework and a interactive UI from Streamlit.
</p>

## Project Structure
```
RAG-Project                 # Root directory
├── streamlit_app.py        # Streamlit application
├── ingest.py               # Document loading, chunking & embedding
├── utils.py                # Load embeddings model, Load vector store
├── query.py                # LangChain RAG retrieval
└── requirements.txt
```

## 🏗️ Architecture
Will add a diagram.



## ✨ Features
- **Interactive UI** — Clean, real-time chat interface built with Streamlit
- **Fast Inference** — Groq's LPU delivers ultra low-latency LLM responses
- **Semantic Search** — HuggingFace embeddings + Pinecone vector retrieval
- **Full Observability** — End-to-end tracing and evaluation via LangSmith
- **Modular Design** — Easily swap out the LLM, embedder, or vector store

## 🛠️ Tech Stack
| Component | Technology | Purpose | 
| --- | -- | --- |
| Frontend | [Streamlit](https://streamlit.io/) | Interactive User Interface |
| Orchestration | [LangChain](https://langchain.com) | Chain composition & RAG logic |
| Observability | [LangSmith](https://smith.langchain.com) | Tracing, debugging & evaluation |
| Vector Store | [Pinecone](https://pinecone.io) | Storing & querying embeddings |
| Inference | [Groq](https://groq.com) | LLM inference |
| Embeddings | [HuggingFace](https://huggingface.co) | Document & query embeddings |
| Language | Python 3.11+ | Core runtime

## 🔮 Next steps
- [ ] Scale up the project to implement multi-user feature
- [ ] Research & experiment with different chunking strategies
- [ ] Add conversation memory for follow-up questions

## 📬 Contact

Made by **Yuhui Cao** — feel free to reach out!

- **GitHub**: [yuhuicaoo](https://github.com/yuhuicaoo)
- **LinkedIn**: [Yuhui Cao](https://www.linkedin.com/in/yuhuicao/)
- **Email**: yuhuicao20@gmail.com

