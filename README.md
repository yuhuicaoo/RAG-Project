# 🔍 RAG-Project
Beginner rag project using LangChain

## Table of Contents
- [Purpose](#purpose)
- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)

## 🎯 Purpose
As a student constantly navigating multiple dense (and often very technical) research papers, I started this project to make it easier to ask questions directly against a paper and get accurate, grounded answers, rather than skimming endlessly or losing context across long documents.


## 🧠 Overview


<p align="justify">
This project implements an end-to-end Retrieval Augmented Generation (RAG) pipeline that enables intelligent question-answering over custom document corpora. Documents are embedded using a HuggingFace embedding model, stored and retrieved from a Pinecone vector database, and responses are generated using Llama4 Scout (via Groq). Built on the LangChain framework and a interactive UI from Streamlit.</p>

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

## 📬 Contact

Made by **Yuhui Cao** — feel free to reach out!

- **GitHub**: [yuhuicaoo](https://github.com/yuhuicaoo)
- **LinkedIn**: [Yuhui Cao](https://www.linkedin.com/in/yuhuicao/)
- **Email**: yuhuicao20@gmail.com

