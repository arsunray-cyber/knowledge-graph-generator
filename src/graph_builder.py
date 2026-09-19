"""
Graph Builder Module
Builds knowledge graph from extracted entities and relationships
"""

import networkx as nx
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict


class GraphBuilder:
    """Build and manage knowledge graph"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Allow multiple edges between nodes
        self.nodes_data = {}
        self.edges_data = []
        
    def add_extraction(self, extraction: Dict[str, Any]) -> None:
        """
        Add extracted entities and relationships to the graph
        
        Args:
            extraction: Dict with 'nodes' and 'relationships' lists
        """
        nodes = extraction.get('nodes', [])
        relationships = extraction.get('relationships', [])
        
        # Add nodes
        for node in nodes:
            self.add_node(
                node_id=node.get('id'),
                label=node.get('label', 'Entity'),
                properties=node.get('properties', {})
            )
        
        # Add relationships
        for rel in relationships:
            self.add_relationship(
                source=rel.get('source'),
                target=rel.get('target'),
                rel_type=rel.get('type', 'RELATED_TO'),
                properties=rel.get('properties', {})
            )
    
    def add_node(self, node_id: str, label: str = 'Entity', 
                 properties: Optional[Dict] = None) -> None:
        """
        Add a node to the graph
        
        Args:
            node_id: Unique identifier for the node
            label: Node type/label
            properties: Additional node properties
        """
        if not node_id:
            return
            
        # Normalize node_id
        node_id = self._normalize_id(node_id)
        
        if node_id not in self.graph:
            self.graph.add_node(
                node_id,
                label=label,
                **(properties or {})
            )
            self.nodes_data[node_id] = {
                'id': node_id,
                'label': label,
                'properties': properties or {},
                'count': 1  # Track how many times this node appears
            }
        else:
            # Node exists, update count and merge properties
            self.nodes_data[node_id]['count'] += 1
            existing_props = self.nodes_data[node_id]['properties']
            existing_props.update(properties or {})
            
            # Update graph node attributes
            self.graph.nodes[node_id]['count'] = self.nodes_data[node_id]['count']
    
    def add_relationship(self, source: str, target: str, 
                        rel_type: str = 'RELATED_TO',
                        properties: Optional[Dict] = None) -> None:
        """
        Add a relationship (edge) to the graph
        
        Args:
            source: Source node ID
            target: Target node ID
            rel_type: Relationship type
            properties: Additional relationship properties
        """
        if not source or not target:
            return
            
        # Normalize IDs
        source = self._normalize_id(source)
        target = self._normalize_id(target)
        
        # Ensure nodes exist
        if source not in self.graph:
            self.add_node(source, 'Entity')
        if target not in self.graph:
            self.add_node(target, 'Entity')
        
        # Add edge
        edge_data = {
            'type': rel_type,
            **(properties or {})
        }
        
        self.graph.add_edge(source, target, **edge_data)
        self.edges_data.append({
            'source': source,
            'target': target,
            'type': rel_type,
            'properties': properties or {}
        })
    
    def _normalize_id(self, node_id: str) -> str:
        """Normalize node ID to be graph-friendly"""
        # Remove special characters, convert to lowercase
        normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in str(node_id))
        # Ensure it doesn't start with a number
        if normalized and normalized[0].isdigit():
            normalized = 'n_' + normalized
        return normalized[:100]  # Limit length
    
    def merge_extractions(self, extractions: List[Dict[str, Any]]) -> None:
        """
        Merge multiple extractions into the graph
        
        Args:
            extractions: List of extraction results
        """
        for extraction in extractions:
            self.add_extraction(extraction)
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the graph"""
        return list(self.nodes_data.values())
    
    def get_edges(self) -> List[Dict[str, Any]]:
        """Get all edges in the graph"""
        edges = []
        seen = set()
        
        for u, v, data in self.graph.edges(data=True):
            edge_key = f"{u}_{v}_{data.get('type', 'RELATED_TO')}"
            if edge_key not in seen:
                edges.append({
                    'source': u,
                    'target': v,
                    'type': data.get('type', 'RELATED_TO'),
                    'properties': {k: v for k, v in data.items() if k != 'type'}
                })
                seen.add(edge_key)
        
        return edges
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'node_types': len(set(d.get('label', 'Entity') for n, d in self.graph.nodes(data=True))),
            'relationship_types': len(set(d.get('type', 'RELATED_TO') for u, v, d in self.graph.edges(data=True))),
            'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
        }
    
    def get_subgraph(self, node_ids: List[str]) -> 'GraphBuilder':
        """
        Extract a subgraph containing only specified nodes
        
        Args:
            node_ids: List of node IDs to include
            
        Returns:
            New GraphBuilder with subgraph
        """
        subgraph = GraphBuilder()
        subgraph.graph = self.graph.subgraph(node_ids).copy()
        
        # Rebuild nodes_data and edges_data
        for node_id in node_ids:
            if node_id in self.nodes_data:
                subgraph.nodes_data[node_id] = self.nodes_data[node_id].copy()
        
        for edge in self.get_edges():
            if edge['source'] in node_ids and edge['target'] in node_ids:
                subgraph.edges_data.append(edge.copy())
        
        return subgraph
    
    def find_connected_components(self) -> List[List[str]]:
        """Find connected components in the graph"""
        # Convert to undirected for component finding
        undirected = self.graph.to_undirected()
        return [list(component) for component in nx.connected_components(undirected)]
    
    def get_central_nodes(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Get most central nodes based on degree centrality
        
        Args:
            top_k: Number of top nodes to return
            
        Returns:
            List of node data sorted by centrality
        """
        if self.graph.number_of_nodes() == 0:
            return []
        
        centrality = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        result = []
        for node_id, score in sorted_nodes[:top_k]:
            node_data = self.nodes_data.get(node_id, {}).copy()
            node_data['centrality_score'] = score
            result.append(node_data)
        
        return result
    
    def clear(self) -> None:
        """Clear the graph"""
        self.graph.clear()
        self.nodes_data.clear()
        self.edges_data.clear()
