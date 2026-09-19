"""
Knowledge Graph Generator - Main Orchestrator Module
Coordinates all components to create knowledge graphs from data sources
"""

import os
import yaml
from typing import List, Dict, Any, Optional
from pathlib import Path

from .data_sources import DataSourceLoader
from .chunking import get_chunker
from .vector_store import VectorStoreManager
from .llm_provider import LLMProvider
from .graph_builder import GraphBuilder
from .visualizer import GraphVisualizer
from .neo4j_exporter import Neo4jExporter


class KnowledgeGraphGenerator:
    """
    Main class for generating knowledge graphs
    
    Usage:
        kg = KnowledgeGraphGenerator(config_path="config/config.yaml")
        kg.add_data_source("website", url="https://example.com")
        kg.chunk_data(strategy="semantic")
        kg.vectorize()
        kg.extract_entities_and_relationships()
        kg.generate_graph()
        kg.visualize()
        kg.export_to_neo4j()
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the Knowledge Graph Generator
        
        Args:
            config_path: Path to YAML configuration file
            config: Direct configuration dictionary (overrides config_path)
        """
        self.config = config or self._load_config(config_path)
        
        # Initialize components
        self.data_loader = DataSourceLoader(self.config.get('data_sources', {}))
        self.vector_store = VectorStoreManager({
            **self.config.get('vector_db', {}),
            'embedding': self.config.get('embedding', {})
        })
        self.llm = LLMProvider(self.config.get('llm', {}))
        self.graph_builder = GraphBuilder()
        self.visualizer = GraphVisualizer(self.config.get('visualization', {}))
        self.neo4j_exporter = Neo4jExporter(self.config.get('neo4j', {}))
        
        # State tracking
        self.chunks = []
        self.extractions = []
        self.is_vectorized = False
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Return default config if file not found
        return {
            'llm': {
                'provider': 'ollama',
                'model': 'llama2',
                'temperature': 0.7,
                'max_tokens': 2000
            },
            'vector_db': {
                'provider': 'qdrant',
                'qdrant': {
                    'host': 'localhost',
                    'port': 6333,
                    'collection_name': 'knowledge_graph_chunks'
                }
            },
            'embedding': {
                'provider': 'sentence-transformers',
                'model': 'all-MiniLM-L6-v2'
            },
            'chunking': {
                'default_strategy': 'semantic'
            },
            'neo4j': {
                'uri': 'bolt://localhost:7687',
                'username': 'neo4j',
                'password': os.getenv('NEO4J_PASSWORD', 'password')
            },
            'visualization': {
                'output_file': 'knowledge_graph.html'
            }
        }
    
    def add_data_source(self, source_type: str, **kwargs) -> 'KnowledgeGraphGenerator':
        """
        Add a data source
        
        Args:
            source_type: Type of source ('website', 'document', 'database', 'text')
            **kwargs: Source-specific parameters
            
        Returns:
            Self for method chaining
        """
        print(f"Loading data from {source_type}...")
        self.data_loader.load(source_type, **kwargs)
        print(f"Loaded {len(self.data_loader.loaded_data)} document(s)")
        return self
    
    def chunk_data(self, strategy: Optional[str] = None, 
                   config: Optional[Dict] = None) -> 'KnowledgeGraphGenerator':
        """
        Chunk the loaded data
        
        Args:
            strategy: Chunking strategy ('semantic', 'paragraph', 'page', 'fixed_size')
            config: Strategy-specific configuration
            
        Returns:
            Self for method chaining
        """
        if not self.data_loader.loaded_data:
            raise ValueError("No data loaded. Call add_data_source() first.")
        
        strategy = strategy or self.config.get('chunking', {}).get('default_strategy', 'semantic')
        config = config or self.config.get('chunking', {}).get(strategy, {})
        
        print(f"Chunking data using '{strategy}' strategy...")
        chunker = get_chunker(strategy, config)
        
        self.chunks = []
        for doc in self.data_loader.loaded_data:
            chunks = chunker.chunk(doc['content'], doc.get('metadata', {}))
            self.chunks.extend(chunks)
        
        print(f"Created {len(self.chunks)} chunks")
        return self
    
    def vectorize(self) -> 'KnowledgeGraphGenerator':
        """
        Vectorize the chunks and store in vector database
        
        Returns:
            Self for method chaining
        """
        if not self.chunks:
            raise ValueError("No chunks available. Call chunk_data() first.")
        
        print("Vectorizing chunks...")
        self.vector_store.add_documents(self.chunks)
        self.is_vectorized = True
        print("Vectorization complete")
        return self
    
    def search_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chunks
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar chunks with scores
        """
        if not self.is_vectorized:
            self.vectorize()
        
        return self.vector_store.similarity_search(query, k=k)
    
    def extract_entities_and_relationships(self, 
                                          use_batch: bool = True,
                                          batch_size: int = 5) -> 'KnowledgeGraphGenerator':
        """
        Extract entities and relationships from chunks using LLM
        
        Args:
            use_batch: Whether to process in batches
            batch_size: Batch size for processing
            
        Returns:
            Self for method chaining
        """
        if not self.chunks:
            raise ValueError("No chunks available. Call chunk_data() first.")
        
        print("Extracting entities and relationships...")
        texts = [chunk['content'] for chunk in self.chunks]
        
        if use_batch:
            self.extractions = self.llm.batch_extract(texts, batch_size=batch_size)
        else:
            self.extractions = [
                self.llm.extract_entities_and_relationships(text)
                for text in texts
            ]
        
        total_nodes = sum(len(e.get('nodes', [])) for e in self.extractions)
        total_rels = sum(len(e.get('relationships', [])) for e in self.extractions)
        
        print(f"Extraction complete: {total_nodes} nodes, {total_rels} relationships")
        return self
    
    def generate_graph(self) -> 'KnowledgeGraphGenerator':
        """
        Build the knowledge graph from extractions
        
        Returns:
            Self for method chaining
        """
        if not self.extractions:
            raise ValueError("No extractions available. Call extract_entities_and_relationships() first.")
        
        print("Building knowledge graph...")
        self.graph_builder.merge_extractions(self.extractions)
        
        stats = self.graph_builder.get_graph_stats()
        print(f"Graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        
        return self
    
    def visualize(self, output_path: Optional[str] = None, 
                  use_plotly: bool = False) -> str:
        """
        Visualize the knowledge graph
        
        Args:
            output_path: Custom output path for visualization
            use_plotly: Use Plotly instead of pyvis
            
        Returns:
            Path to generated visualization file
        """
        if self.graph_builder.graph.number_of_nodes() == 0:
            raise ValueError("No graph to visualize. Call generate_graph() first.")
        
        print("Generating visualization...")
        
        if use_plotly:
            return self.visualizer.visualize_with_plotly(
                self.graph_builder, 
                output_path
            )
        else:
            return self.visualizer.visualize(
                self.graph_builder, 
                output_path
            )
    
    def export_to_neo4j(self, connect: bool = True) -> Dict[str, int]:
        """
        Export the graph to Neo4j database
        
        Args:
            connect: Whether to establish connection (if False, must call connect first)
            
        Returns:
            Dictionary with counts of created nodes and relationships
        """
        if self.graph_builder.graph.number_of_nodes() == 0:
            raise ValueError("No graph to export. Call generate_graph() first.")
        
        print("Exporting to Neo4j...")
        
        if connect:
            if not self.neo4j_exporter.connect():
                return {'nodes_created': 0, 'relationships_created': 0}
        
        result = self.neo4j_exporter.export_graph(self.graph_builder)
        
        print(f"Export complete: {result['nodes_created']} nodes, "
              f"{result['relationships_created']} relationships created")
        
        return result
    
    def generate_cypher_script(self, output_path: Optional[str] = None) -> str:
        """
        Generate Cypher script for Neo4j
        
        Args:
            output_path: Path to save the Cypher script
            
        Returns:
            Generated Cypher script
        """
        if self.graph_builder.graph.number_of_nodes() == 0:
            raise ValueError("No graph to export. Call generate_graph() first.")
        
        return self.neo4j_exporter.generate_cypher(
            self.graph_builder, 
            output_path
        )
    
    def get_stats_report(self) -> str:
        """Get statistics report about the graph"""
        if self.graph_builder.graph.number_of_nodes() == 0:
            return "No graph data available"
        
        return self.visualizer.generate_stats_report(self.graph_builder)
    
    def get_central_nodes(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get most central nodes in the graph"""
        return self.graph_builder.get_central_nodes(top_k=top_k)
    
    def clear(self):
        """Clear all state"""
        self.data_loader.clear()
        self.chunks = []
        self.extractions = []
        self.graph_builder.clear()
        self.is_vectorized = False
        print("All state cleared")
    
    def run_pipeline(self, source_type: str, 
                    chunk_strategy: str = 'semantic',
                    visualize: bool = True,
                    export_neo4j: bool = False,
                    **source_kwargs) -> Dict[str, Any]:
        """
        Run the complete pipeline from data source to graph
        
        Args:
            source_type: Type of data source
            chunk_strategy: Chunking strategy to use
            visualize: Whether to create visualization
            export_neo4j: Whether to export to Neo4j
            **source_kwargs: Arguments for the data source
            
        Returns:
            Dictionary with results and paths
        """
        results = {}
        
        # Run pipeline
        self.add_data_source(source_type, **source_kwargs)
        self.chunk_data(strategy=chunk_strategy)
        self.vectorize()
        self.extract_entities_and_relationships()
        self.generate_graph()
        
        results['stats'] = self.graph_builder.get_graph_stats()
        results['central_nodes'] = self.get_central_nodes(top_k=5)
        
        if visualize:
            results['visualization_path'] = self.visualize()
        
        if export_neo4j:
            results['neo4j_export'] = self.export_to_neo4j()
        
        return results
