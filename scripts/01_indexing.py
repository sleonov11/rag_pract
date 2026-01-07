import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from typing import Any, List, Dict, Tuple
import docx2txt

class SimpleRAGIndexer:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Инициализация индексатора

        Args:
            model_name: название модели для эмбеддингов
         """

        self.model = SentenceTransformer(model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            separators=['\n\n', '\n', ' ', '']
        )
        self.index = None
        self.chunks = []
        self.metadata = []

    def load_documents(self, folder_path):
        """
        Загрузка документов из папки

        Args: folder_path: путь к папке с документами
        """

        documents = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            if filename.endswith('.txt'):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        documents.append({
                            'text': text,
                            'source': filename
                        })
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
            elif filename.endswith('.pdf'):
                try:
                    reader = PdfReader(file_path)
                    text_pages = []
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_pages.append(page_text)
                    text = '\n'.join(text_pages)
                    documents.append({
                        'text': text,
                        'source':filename
                    })
                except Exception as e:
                    print(f"Error loading PDF {filename}: {e}")

            elif filename.endswith('.docx'):
                try:
                    text = docx2txt.process(file_path)
                    documents.append({
                        'text': text,
                        'source': filename
                    })
                except Exception as e:
                    print(f"Error loading docx {filename}: {e}")
            else:
                print(f"Unsupported format: {filename}")
        return documents

    def split_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        all_chunks = []
        all_metadata = []

        for doc in documents:
            chunks = self.text_splitter.split_text(doc['text'])

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'source': doc['source'],
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
        return all_chunks, all_metadata

    def create_index(self, chunks):
        """
        creation of faiss indexes
        :param chunks:
        :return: embeddings
        """

        print(f"Создание эмбеддингов для {len(chunks)} чанков...")
        embeddings = self.model.encode(chunks, show_progress_bar = True)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension) # L2 расстояние (евклидово)

        self.index.add(embeddings.astype('float32'))
        print(f"Индекс создан. Размерность: {dimension}, кол-во векторов:"
              f"{self.index.ntotal}")
        return embeddings

    def build(self, documents_folder):
        """
            Полный пайплайн индексирования

            Args:
                documents_folder: путь к папке с документами
        """
        print("1. Загрузка документов...")
        documents = self.load_documents(documents_folder)
        print(f"   Загружено документов: {len(documents)}")

        print("2. Разбиение на чанки...")
        self.chunks, self.metadata = self.split_documents(documents)
        print(f"   Создано чанков: {len(self.chunks)}")

        print("3. Создание векторного индекса...")
        embeddings = self.create_index(self.chunks)

        print("4. Сохранение индекса и данных...")
        self.save('rag_index')

        return {
            'num_documents': len(documents),
            'num_chunks': len(self.chunks),
            'embedding_dim': embeddings.shape[1]
        }

    def save(self, output_path):
        # Сохраняем FAISS индекс
        faiss.write_index(self.index, f"{output_path}.faiss")
        # Сохраняем чанки и метаданные
        with open(f"{output_path}_data.pkl", "wb") as f:
            pickle.dump({
                'chunks':self.chunks,
                'metadata': self.metadata,
                'model_name': self.model.get_sentence_embedding_dimension()
            }, f)
        print(f"Индекс сохранен: {output_path}.faiss")
        print(f"Данные сохранены: {output_path}_data.pkl")

    def load(self, input_path):
        # Загрузка индекса и данных

        self.index = faiss.read_index(f"{input_path}.faiss")
        with open(f"{input_path}_data.pkl", 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadata = data['metadata']

        print(f"Загружено {len(self.chunks)} чанков, индекс размером {self.index.ntotal} векторов")


if __name__ == "__main__":
    indexer = SimpleRAGIndexer()
    docs_folder = 'D:/rag_pract/data'
    stats = indexer.build(docs_folder)
    print("\n Индексирование завершено!")
    print(f"   Документов: {stats['num_documents']}")
    print(f"   Чанков: {stats['num_chunks']}")
    print(f"   Размерность эмбеддингов: {stats['embedding_dim']}")