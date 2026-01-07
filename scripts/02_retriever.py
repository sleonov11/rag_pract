import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict, Any
import os

class SimpleRAGRetriever:
    def __init__(self, index_path:str,
                 model_name:str = 'all-MiniLM-L6-v2',
                 cache_folder='./model_cache'):
        """
            Инициализация ретривера

            Args:
                index_path: путь к сохраненному индексу (без расширения)
                model_name: название модели для эмбеддингов
        """
        self.index = faiss.read_index(f"{index_path}.faiss")
        with open(f"{index_path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']

        self.model = SentenceTransformer(model_name, cache_folder=cache_folder)
        print(f"Ретривер загружен. Чанков: {len(self.chunks)},"
              f" векторов: {self.index.ntotal}")

    def retrieve(self, query: str, top_k: int = 5)-> List[Dict[str, Any]]:
        query_embeding = self.model.encode([query])
         #search indexes faiss
        distances, indices = self.index.search(query_embeding.astype('float32'), top_k)

        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:
                results.append({
                    'chunk':self.chunks[idx],
                    'metadata': self.metadata[idx],
                    'score': float(distance),
                    'similarity': 1 / (1+distance)
                })
        return results

    def retrieve_with_threshold(self,
                                query: str,
                                top_k:int = 10,
                                similarity_threshold: float = 0.7)->List[Dict[str, Any]]:
        all_results = self.retrieve(query, top_k)
        filtered_results = [
            result for result in all_results
            if result['similarity'] >= similarity_threshold
        ]

        return filtered_results


if __name__ == '__main__':
    pass