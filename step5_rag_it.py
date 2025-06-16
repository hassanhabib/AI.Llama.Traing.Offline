# rag_llama_local.py

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from llama_cpp import Llama
import pyodbc
import os

# Step 1: Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 2: Connect to MSSQL Developer Edition using Windows Authentication
connection_string = (
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=BIGB;"
    "DATABASE=LlamaDB;"
    "Trusted_Connection=yes;"
)
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

# Assume we have a table 'knowledge' with a 'content' column
cursor.execute("SELECT content FROM knowledge")
documents = [row[0] for row in cursor.fetchall() if row[0] is not None]

# Step 3: Embed documents
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# Step 4: Build FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# Map index to original text
id_to_doc = {i: doc for i, doc in enumerate(documents)}

# Step 5: Set up LLaMA
model_path = "C:/Users/hassa/OneDrive/Desktop/AI Resources/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"LLaMA model file not found at: {model_path}")

llm = Llama(model_path=model_path, n_ctx=2048, n_threads=4)

# Step 6: Query function
def query_rag(question, top_k=3):
    # Embed the query
    query_vec = embedder.encode([question], convert_to_numpy=True)

    # Search FAISS index
    distances, indices = index.search(query_vec, top_k)
    retrieved_docs = [id_to_doc[idx] for idx in indices[0]]

    # Combine context
    context = "\n\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    # Query LLaMA
    response = llm(prompt, max_tokens=200)
    answer = response['choices'][0]['text'].strip()

    return answer

# Example usage
if __name__ == "__main__":
    question = "What are company hours?"
    answer = query_rag(question)
    print("Answer:", answer)

    # Close DB connection
    conn.close()
