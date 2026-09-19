#!/usr/bin/env python3
"""
Example usage of the Knowledge Graph Generator

This script demonstrates how to use the knowledge graph generator
with different data sources and configurations.
"""

import os
from pathlib import Path

# Import the main class
from src.knowledge_graph import KnowledgeGraphGenerator


def example_with_text():
    """Example: Create knowledge graph from text input"""
    
    # Sample text about technology companies
    sample_text = """
    Apple Inc. is an American multinational technology company headquartered 
    in Cupertino, California. Apple was founded by Steve Jobs, Steve Wozniak, 
    and Ronald Wayne in April 1976. The company develops, manufactures, and 
    sells consumer electronics, computer software, and online services.
    
    Tim Cook is the current CEO of Apple, having succeeded Steve Jobs in 2011.
    Under Tim Cook's leadership, Apple has continued to innovate with products
    like the Apple Watch and AirPods.
    
    Microsoft Corporation is another major technology company based in Redmond,
    Washington. It was founded by Bill Gates and Paul Allen in 1975. Microsoft
    is best known for its Windows operating system and Office productivity suite.
    
    Satya Nadella became CEO of Microsoft in 2014, transforming the company's
    focus towards cloud computing with Azure. Microsoft also owns LinkedIn and GitHub.
    
    Google, now part of Alphabet Inc., was founded by Larry Page and Sergey Brin
    while they were PhD students at Stanford University. Google's headquarters,
    known as the Googleplex, is located in Mountain View, California.
    """
    
    print("=" * 60)
    print("EXAMPLE 1: Knowledge Graph from Text")
    print("=" * 60)
    
    # Initialize with default config
    kg = KnowledgeGraphGenerator()
    
    # Add text as data source
    kg.add_data_source("text", text=sample_text)
    
    # Chunk the data (using semantic chunking)
    kg.chunk_data(strategy="semantic")
    
    # Vectorize chunks
    kg.vectorize()
    
    # Extract entities and relationships using LLM
    kg.extract_entities_and_relationships()
    
    # Build the graph
    kg.generate_graph()
    
    # Get statistics
    print("\n" + kg.get_stats_report())
    
    # Visualize the graph
    output_path = kg.visualize(output_path="example_text_graph.html")
    print(f"\nVisualization saved to: {output_path}")
    
    # Generate Cypher script for Neo4j
    cypher_script = kg.generate_cypher_script(output_path="example_text.cypher")
    print(f"Cypher script saved to: {cypher_script}")
    
    return kg


def example_with_document(doc_path: str):
    """Example: Create knowledge graph from a document file"""
    
    print("=" * 60)
    print("EXAMPLE 2: Knowledge Graph from Document")
    print("=" * 60)
    
    if not os.path.exists(doc_path):
        print(f"Document not found: {doc_path}")
        print("Skipping document example.")
        return None
    
    kg = KnowledgeGraphGenerator()
    
    # Load from document (supports PDF, DOCX, TXT, MD)
    kg.add_data_source("document", path=doc_path)
    
    # Use paragraph chunking for documents
    kg.chunk_data(strategy="paragraph")
    
    kg.vectorize()
    kg.extract_entities_and_relationships()
    kg.generate_graph()
    
    print(kg.get_stats_report())
    
    kg.visualize(output_path="example_document_graph.html")
    
    return kg


def example_with_website(url: str = "https://en.wikipedia.org/wiki/Artificial_intelligence"):
    """Example: Create knowledge graph from website content"""
    
    print("=" * 60)
    print("EXAMPLE 3: Knowledge Graph from Website")
    print("=" * 60)
    
    kg = KnowledgeGraphGenerator()
    
    # Scrape website content
    kg.add_data_source("website", url=url, max_depth=1, follow_links=False)
    
    # Semantic chunking works well for web content
    kg.chunk_data(strategy="semantic")
    
    kg.vectorize()
    kg.extract_entities_and_relationships()
    kg.generate_graph()
    
    print(kg.get_stats_report())
    
    kg.visualize(output_path="example_website_graph.html")
    
    return kg


def example_custom_config():
    """Example: Using custom configuration"""
    
    print("=" * 60)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 60)
    
    # Custom config dictionary
    custom_config = {
        'llm': {
            'provider': 'ollama',  # Options: 'ollama', 'grok', 'openrouter'
            'model': 'llama2',
            'temperature': 0.5,
            'max_tokens': 1500
        },
        'vector_db': {
            'provider': 'qdrant',  # Options: 'qdrant', 'chroma', 'weaviate', 'pinecone'
            'qdrant': {
                'host': 'localhost',
                'port': 6333,
                'collection_name': 'my_knowledge_graph'
            }
        },
        'embedding': {
            'provider': 'sentence-transformers',
            'model': 'all-MiniLM-L6-v2'
        },
        'chunking': {
            'default_strategy': 'fixed_size',
            'fixed_size': {
                'chunk_size': 300,
                'chunk_overlap': 50
            }
        },
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': 'your_password_here'
        },
        'visualization': {
            'output_file': 'custom_graph.html',
            'height': '800px',
            'width': '100%'
        }
    }
    
    kg = KnowledgeGraphGenerator(config=custom_config)
    
    sample_text = """
    Neural networks are computing systems inspired by biological neural networks.
    They consist of layers of interconnected nodes that process information.
    Deep learning uses multiple layers of neural networks to learn hierarchical
    representations of data.
    """
    
    kg.add_data_source("text", text=sample_text)
    kg.chunk_data()
    kg.vectorize()
    kg.extract_entities_and_relationships()
    kg.generate_graph()
    
    print(kg.get_stats_report())
    
    return kg


def export_to_neo4j_example():
    """Example: Export graph to Neo4j database"""
    
    print("=" * 60)
    print("EXAMPLE 5: Export to Neo4j")
    print("=" * 60)
    
    # Make sure you have Neo4j running
    neo4j_config = {
        'neo4j': {
            'uri': 'bolt://localhost:7687',
            'username': 'neo4j',
            'password': os.getenv('NEO4J_PASSWORD', 'password')
        }
    }
    
    kg = KnowledgeGraphGenerator(config=neo4j_config)
    
    sample_text = """
    Python is a programming language created by Guido van Rossum.
    It emphasizes code readability with significant indentation.
    Python supports multiple programming paradigms including procedural,
    object-oriented, and functional programming.
    """
    
    kg.add_data_source("text", text=sample_text)
    kg.chunk_data()
    kg.vectorize()
    kg.extract_entities_and_relationships()
    kg.generate_graph()
    
    # Export to Neo4j
    result = kg.export_to_neo4j(connect=True)
    print(f"Export result: {result}")
    
    # Or generate Cypher script without connecting
    cypher = kg.generate_cypher_script(output_path="export.cypher")
    print(f"Cypher script generated: {cypher}")
    
    return kg


def main():
    """Run all examples"""
    
    print("\n" + "=" * 60)
    print("KNOWLEDGE GRAPH GENERATOR - EXAMPLES")
    print("=" * 60 + "\n")
    
    # Example 1: Text input (always works)
    try:
        kg1 = example_with_text()
        print("\n✓ Text example completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Text example failed: {e}\n")
    
    # Example 2: Document (if exists)
    # Uncomment and provide a valid path
    # doc_path = "/path/to/your/document.pdf"
    # kg2 = example_with_document(doc_path)
    
    # Example 3: Website (requires internet)
    try:
        kg3 = example_with_website()
        print("\n✓ Website example completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Website example failed: {e}\n")
    
    # Example 4: Custom config
    try:
        kg4 = example_custom_config()
        print("\n✓ Custom config example completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Custom config example failed: {e}\n")
    
    # Example 5: Neo4j export (requires Neo4j running)
    # Uncomment if you have Neo4j running
    # kg5 = export_to_neo4j_example()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print("\nTo visualize your graphs, open the generated HTML files in a browser.")
    print("To load into Neo4j, run the generated .cypher scripts in Neo4j Browser.\n")


if __name__ == "__main__":
    main()
