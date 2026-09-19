"""
Vector Store Manager Module
Supports: Qdrant (primary), Pinecone, Weaviate, Chroma
"""

import os
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class VectorStoreManager:
    """Manage vector storage across different backends"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'qdrant')
        self._store = None
        self._embeddings = None
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        emb_config = self.config.get('embedding', {})
        provider = emb_config.get('provider', 'sentence-transformers')
        model_name = emb_config.get('model', 'all-MiniLM-L6-v2')
        
        if provider == 'sentence-transformers':
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embeddings = HuggingFaceEmbeddings(model_name=model_name)
        elif provider == 'openai':
            from langchain_openai import OpenAIEmbeddings
            api_key = emb_config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
            self._embeddings = OpenAIEmbeddings(api_key=api_key)
        elif provider == 'ollama':
            from langchain_community.embeddings import OllamaEmbeddings
            base_url = emb_config.get('ollama_base_url', 'http://localhost:11434')
            self._embeddings = OllamaEmbeddings(base_url=base_url, model=model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        return self._embeddings
    
    def get_embeddings(self):
        """Get or create embeddings instance"""
        if self._embeddings is None:
            self._initialize_embeddings()
        return self._embeddings
    
    def initialize(self):
        """Initialize the vector store based on provider"""
        if self.provider == 'qdrant':
            self._init_qdrant()
        elif self.provider == 'pinecone':
            self._init_pinecone()
        elif self.provider == 'weaviate':
            self._init_weaviate()
        elif self.provider == 'chroma':
            self._init_chroma()
        else:
            raise ValueError(f"Unsupported vector DB provider: {self.provider}")
        
        return self._store
    
    def _init_qdrant(self):
        """Initialize Qdrant vector store"""
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient
        
        qdrant_config = self.config.get('qdrant', {})
        host = qdrant_config.get('host', 'localhost')
        port = qdrant_config.get('port', 6333)
        https = qdrant_config.get('https', False)
        api_key = qdrant_config.get('api_key') or os.getenv('QDRANT_API_KEY')
        collection_name = qdrant_config.get('collection_name', 'knowledge_graph_chunks')
        vector_size = qdrant_config.get('vector_size', 768)
        
        # Initialize client
        client_params = {
            'url': f"http{'s' if https else ''}://{host}:{port}",
        }
        if api_key:
            client_params['api_key'] = api_key
        
        client = QdrantClient(**client_params)
        
        # Get embeddings to determine vector size
        embeddings = self.get_embeddings()
        
        # Create collection if it doesn't exist
        try:
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            if not collection_exists:
                from qdrant_client.http.models import Distance, VectorParams
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"Warning: Could not check/create collection: {e}")
        
        self._store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        
    def _init_pinecone(self):
        """Initialize Pinecone vector store"""
        from langchain_community.vectorstores import Pinecone
        import pinecone
        
        pinecone_config = self.config.get('pinecone', {})
        api_key = pinecone_config.get('api_key') or os.getenv('PINECONE_API_KEY')
        environment = pinecone_config.get('environment', 'us-west1-gcp')
        index_name = pinecone_config.get('index_name', 'knowledge-graph')
        
        pinecone.init(api_key=api_key, environment=environment)
        
        embeddings = self.get_embeddings()
        
        self._store = Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
    def _init_weaviate(self):
        """Initialize Weaviate vector store"""
        from langchain_community.vectorstores import Weaviate
        import weaviate
        
        weaviate_config = self.config.get('weaviate', {})
        url = weaviate_config.get('url', 'http://localhost:8080')
        api_key = weaviate_config.get('api_key') or os.getenv('WEAVIATE_API_KEY')
        class_name = weaviate_config.get('class_name', 'KnowledgeGraphChunk')
        
        auth_config = None
        if api_key:
            auth_config = weaviate.auth.AuthApiKey(api_key=api_key)
        
        client = weaviate.Client(url=url, auth_client_secret=auth_config)
        
        embeddings = self.get_embeddings()
        
        self._store = Weaviate(
            client=client,
            index_name=class_name,
            text_key="content",
            embedding=embeddings,
            by_text=False,
        )
        
    def _init_chroma(self):
        """Initialize Chroma vector store"""
        from langchain_community.vectorstores import Chroma
        
        chroma_config = self.config.get('chroma', {})
        persist_directory = chroma_config.get('persist_directory', './chroma_db')
        collection_name = chroma_config.get('collection_name', 'knowledge_graph')
        
        embeddings = self.get_embeddings()
        
        self._store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
    
    def add_documents(self, documents: List[Dict[str, Any]], 
                     batch_size: int = 32) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: List of dicts with 'content' and 'metadata'
            batch_size: Number of documents to add at once
            
        Returns:
            List of document IDs
        """
        if self._store is None:
            self.initialize()
        
        from langchain.schema import Document
        
        langchain_docs = []
        for doc in documents:
            langchain_docs.append(Document(
                page_content=doc['content'],
                metadata=doc.get('metadata', {})
            ))
        
        ids = self._store.add_documents(langchain_docs, batch_size=batch_size)
        return ids
    
    def similarity_search(self, query: str, k: int = 5, 
                         filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        if self._store is None:
            self.initialize()
        
        results = self._store.similarity_search_with_score(
            query, 
            k=k,
            filter=filter_dict
        )
        
        return [
            {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'score': float(score)
            }
            for doc, score in results
        ]
    
    def delete_documents(self, ids: List[str]) -> bool:
        """Delete documents by ID"""
        if self._store is None:
            self.initialize()
        
        try:
            self._store.delete(ids)
            return True
        except Exception as e:
            print(f"Error deleting documents: {e}")
            return False
    
    def get_store(self):
        """Get the underlying vector store instance"""
        if self._store is None:
            self.initialize()
        return self._store
