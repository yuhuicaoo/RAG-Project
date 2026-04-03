import os
from query import get_agent
from ingest import ingest_document, document_already_exists
from utils import load_embeddings_model_from_HF, get_vector_store

DOCUMENTS_DIR = "documents"

if __name__ == "__main__":
    print(f"Loading embedding model \n")
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
        print(f"Ingesting documents \n")
        for filename in os.listdir(DOCUMENTS_DIR):
            if filename.endswith(".pdf"):
                file_path = os.path.join(DOCUMENTS_DIR, filename)
                ingest_document(file_path, embedding_model)
    else:
        print(f"All documents already in the database, skipping...")

    print(f"Creating agent \n")
    agent = get_agent(vector_store)

    print("\nRAG is ready! Type 'exit' to quit.\n")
    while True:
        query = input("You: ").strip()

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye")
            break

        print("\n --- Agent Response ---")
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]}, stream_mode="values"
        ):
            event["messages"][-1].pretty_print()
