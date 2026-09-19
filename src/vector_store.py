"""
Neo4j Vector Store Manager

Handles vector indexing and similarity search using Neo4j's native vector indexes.
"""

from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np


class Neo4jVectorStore:
    """
    Manages vector embeddings and similarity search using Neo4j's native vector indexes.
    
    Neo4j provides native vector search capabilities through its vector index feature,
    eliminating the need for external vector databases.
    """
    
    def __init__(self, neo4j_driver, embedding_model: str = "all-MiniLM-L6-v2", 
                 index_name: str = "chunk_embeddings", label: str = "Chunk"):
        """
        Initialize Neo4j Vector Store.
        
        Args:
            neo4j_driver: Neo4j driver instance
            embedding_model: Name of the sentence transformer model to use
            index_name: Name of the vector index in Neo4j
            label: Node label for chunks
        """
        self.driver = neo4j_driver
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_name = index_name
        self.label = label
        # Use the new method name with fallback for older versions
        try:
            self.embedding_dim = self.embedding_model.get_embedding_dimension()
        except AttributeError:
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
    def create_vector_index(self):
        """
        Create a vector index in Neo4j for similarity search.
        """
        with self.driver.session() as session:
            # Check if index already exists
            result = session.run(
                "SHOW INDEXES WHERE name = $index_name",
                index_name=self.index_name
            )
            existing = result.single()
            
            if existing:
                print(f"Vector index '{self.index_name}' already exists.")
                return
            
            # Create vector index
            session.run(f"""
                CREATE VECTOR INDEX `{self.index_name}`
                FOR (n:{self.label})
                ON n.embedding
                OPTIONS {{
                    indexConfig: {{
                        `vector.dimensions`: {self.embedding_dim},
                        `vector.similarity_function`: 'cosine'
                    }}
                }}
            """)
            print(f"Created vector index '{self.index_name}' with dimension {self.embedding_dim}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text chunk.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.embedding_model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def store_chunk(self, chunk_id: str, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a chunk with its embedding in Neo4j.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: Text content of the chunk
            metadata: Optional metadata dictionary
        """
        embedding = self.generate_embedding(text)
        
        with self.driver.session() as session:
            # Create or merge the chunk node with embedding
            query = f"""
                MERGE (c:{self.label} {{id: $chunk_id}})
                SET c.text = $text,
                    c.embedding = $embedding
            """
            
            if metadata:
                for key, value in metadata.items():
                    query += f", c.{key} = ${key}"
            
            session.run(query, chunk_id=chunk_id, text=text, embedding=embedding, **metadata)
    
    def store_chunks_batch(self, chunks: List[Dict[str, Any]]):
        """
        Store multiple chunks with their embeddings in Neo4j.
        
        Args:
            chunks: List of dictionaries containing 'id', 'text', and optional 'metadata'
        """
        with self.driver.session() as session:
            for chunk in chunks:
                chunk_id = chunk['id']
                text = chunk['text']
                metadata = chunk.get('metadata', {})
                
                embedding = self.generate_embedding(text)
                
                # Build dynamic properties
                props = {
                    'id': chunk_id,
                    'text': text,
                    'embedding': embedding
                }
                props.update(metadata)
                
                # Create node with all properties
                set_clauses = []
                params = {}
                for i, (key, value) in enumerate(props.items()):
                    param_name = f"param_{i}"
                    set_clauses.append(f"c.{key} = ${param_name}")
                    params[param_name] = value
                
                query = f"""
                    MERGE (c:{self.label} {{id: $chunk_id}})
                    SET {', '.join(set_clauses)}
                """
                params['chunk_id'] = chunk_id
                
                session.run(query, **params)
    
    def similarity_search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform similarity search using vector index.
        
        Args:
            query_text: Query text to search for similar chunks
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing matching chunks and their scores
        """
        query_embedding = self.generate_embedding(query_text)
        
        with self.driver.session() as session:
            result = session.run(f"""
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_embedding)
                YIELD node AS chunk, score
                RETURN chunk.id AS id, chunk.text AS text, score
                ORDER BY score DESC
            """, index_name=self.index_name, top_k=top_k, query_embedding=query_embedding)
            
            matches = []
            for record in result:
                matches.append({
                    'id': record['id'],
                    'text': record['text'],
                    'score': record['score']
                })
            
            return matches
    
    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """
        Retrieve all chunks from the database.
        
        Returns:
            List of all chunk dictionaries
        """
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (c:{self.label})
                RETURN c.id AS id, c.text AS text, properties(c) AS metadata
            """)
            
            chunks = []
            for record in result:
                metadata = dict(record['metadata'])
                metadata.pop('embedding', None)  # Exclude embedding from metadata
                chunks.append({
                    'id': record['id'],
                    'text': record['text'],
                    'metadata': metadata
                })
            
            return chunks
    
    def delete_index(self):
        """
        Delete the vector index.
        """
        with self.driver.session() as session:
            session.run(f"DROP INDEX `{self.index_name}` IF EXISTS")
            print(f"Deleted vector index '{self.index_name}'")
    
    def close(self):
        """
        Close the Neo4j driver connection.
        """
        if self.driver:
            self.driver.close()
