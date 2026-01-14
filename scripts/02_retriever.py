import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict, Any
import os
from transformers import pipeline

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

from transformers import pipeline

class RAGGenerator:
    def __init__(self, generator_model_name="facebook/bart-large-cnn"):
        self.summarizer = pipeline("summarization", model=generator_model_name)

    def generate_answer(self, context: str, question: str) -> str:
        combined_input = f"Question: {question}\nContext: {context}"
        summary = self.summarizer(combined_input, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']

if __name__ == '__main__':
    retriever = SimpleRAGRetriever(index_path='rag_index')
    query = "структуры данных "
    results = retriever.retrieve(query, top_k=5)

    for i, res in enumerate(results):
        print(f"\n--- Результат {i + 1} ---")
        print(f"Источник: {res['metadata']['source']}")
        print(f"Содержимое: {res['chunk'][:200]}...")  # первые 200 символов
        print(f"Схожесть: {res['similarity']:.4f}")

    context = "\n".join([res['chunk'] for res in results])

    # Создаем объект генератора
    generator = RAGGenerator()

    # Вызываем метод у объекта
    answer = generator.generate_answer(context, query)
    print("\nСгенерированный ответ:", answer)