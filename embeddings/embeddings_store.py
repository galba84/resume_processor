import chromadb
import numpy as np

class EmbeddingsStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name="resume_embeddings")

    def add_embeddings(self, embeddings, ids, documents):
        self.collection.add(embeddings=embeddings, ids=ids, documents=documents)

    def search(self, query_embedding, k=5):
        results = self.collection.query(query_embeddings=[query_embedding], n_results=k)
        return results['ids'][0]
