"""
Neo4j Exporter Module
Generates Cypher queries and exports to Neo4j database
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class Neo4jExporter:
    """Export knowledge graph to Neo4j database"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.uri = config.get('uri', 'bolt://localhost:7687')
        self.username = config.get('username', 'neo4j')
        self.password = config.get('password') or os.getenv('NEO4J_PASSWORD', 'password')
        self.database = config.get('database', 'neo4j')
        self._driver = None
        
    def connect(self) -> bool:
        """Establish connection to Neo4j"""
        try:
            from neo4j import GraphDatabase
            
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                database=self.database
            )
            
            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")
            
            print(f"Connected to Neo4j at {self.uri}")
            return True
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            return False
    
    def disconnect(self):
        """Close Neo4j connection"""
        if self._driver:
            self._driver.close()
            print("Neo4j connection closed")
    
    def export_graph(self, graph_builder, batch_size: int = 100) -> Dict[str, int]:
        """
        Export entire graph to Neo4j
        
        Args:
            graph_builder: GraphBuilder instance
            batch_size: Number of nodes/edges to process in each batch
            
        Returns:
            Dictionary with counts of created nodes and relationships
        """
        if not self._driver:
            if not self.connect():
                return {'nodes_created': 0, 'relationships_created': 0}
        
        nodes = graph_builder.get_nodes()
        edges = graph_builder.get_edges()
        
        nodes_created = 0
        relationships_created = 0
        
        # Create nodes
        print(f"Creating {len(nodes)} nodes...")
        for node in nodes:
            if self._create_node(node):
                nodes_created += 1
        
        # Create relationships
        print(f"Creating {len(edges)} relationships...")
        for edge in edges:
            if self._create_relationship(edge):
                relationships_created += 1
        
        return {
            'nodes_created': nodes_created,
            'relationships_created': relationships_created
        }
    
    def _create_node(self, node: Dict[str, Any]) -> bool:
        """Create a single node in Neo4j"""
        try:
            from neo4j.exceptions import CypherSyntaxError
            
            node_id = node['id']
            label = node.get('label', 'Entity')
            properties = node.get('properties', {})
            
            # Sanitize label (Neo4j labels can't have spaces or special chars)
            label = self._sanitize_label(label)
            
            query = f"""
            MERGE (n:{label} {{id: $id}})
            SET n += $properties,
                n.name = $name,
                n.updated_at = datetime()
            RETURN n
            """
            
            with self._driver.session() as session:
                session.run(
                    query,
                    id=node_id,
                    name=properties.get('name', node_id),
                    properties={
                        k: v for k, v in properties.items() 
                        if k not in ['name'] and isinstance(v, (str, int, float, bool))
                    }
                )
            return True
        except Exception as e:
            print(f"Error creating node {node_id}: {e}")
            return False
    
    def _create_relationship(self, edge: Dict[str, Any]) -> bool:
        """Create a single relationship in Neo4j"""
        try:
            source = edge['source']
            target = edge['target']
            rel_type = edge.get('type', 'RELATED_TO')
            properties = edge.get('properties', {})
            
            # Sanitize relationship type
            rel_type = self._sanitize_rel_type(rel_type)
            
            # Query to find nodes and create relationship
            query = f"""
            MATCH (source), (target)
            WHERE source.id = $source_id OR source.name = $source_id
            AND target.id = $target_id OR target.name = $target_id
            MERGE (source)-[r:{rel_type}]->(target)
            SET r += $properties,
                r.created_at = datetime()
            RETURN r
            """
            
            with self._driver.session() as session:
                result = session.run(
                    query,
                    source_id=source,
                    target_id=target,
                    properties={
                        k: v for k, v in properties.items()
                        if isinstance(v, (str, int, float, bool))
                    }
                )
                result.consume()  # Consume the result
            
            return True
        except Exception as e:
            print(f"Error creating relationship {source}->{target}: {e}")
            return False
    
    def _sanitize_label(self, label: str) -> str:
        """Sanitize label for Neo4j"""
        # Remove special characters, replace spaces with underscores
        sanitized = ''.join(c if c.isalnum() else '_' for c in label)
        # Ensure it starts with a letter
        if sanitized and sanitized[0].isdigit():
            sanitized = 'N_' + sanitized
        return sanitized[:50]  # Limit length
    
    def _sanitize_rel_type(self, rel_type: str) -> str:
        """Sanitize relationship type for Neo4j"""
        # Relationship types should be uppercase with underscores
        sanitized = rel_type.upper().replace(' ', '_').replace('-', '_')
        # Remove special characters
        sanitized = ''.join(c if c.isalnum() or c == '_' else '' for c in sanitized)
        return sanitized[:50]
    
    def generate_cypher(self, graph_builder, output_path: Optional[str] = None) -> str:
        """
        Generate Cypher script for creating the graph
        
        Args:
            graph_builder: GraphBuilder instance
            output_path: Optional path to save the Cypher script
            
        Returns:
            Generated Cypher script
        """
        nodes = graph_builder.get_nodes()
        edges = graph_builder.get_edges()
        
        cypher_lines = [
            "// Knowledge Graph Cypher Export",
            f"// Generated: {datetime.now().isoformat()}",
            "// Use this script with Neo4j Browser or cypher-shell",
            "",
            "// Create constraints for unique IDs (run once)",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE;",
            ""
        ]
        
        # Generate node creation statements
        cypher_lines.append("// Create Nodes")
        for node in nodes:
            node_id = node['id']
            label = self._sanitize_label(node.get('label', 'Entity'))
            properties = node.get('properties', {})
            name = properties.get('name', node_id)
            
            props_str = ", ".join([
                f"{k}: '{self._escape_string(v)}'" 
                for k, v in properties.items() 
                if isinstance(v, str) and k != 'name'
            ])
            
            if props_str:
                cypher_lines.append(
                    f"MERGE (n:{label} {{id: '{node_id}'}}) "
                    f"SET n.name = '{self._escape_string(name)}', "
                    f"n += {{{props_str}}};"
                )
            else:
                cypher_lines.append(
                    f"MERGE (n:{label} {{id: '{node_id}'}}) "
                    f"SET n.name = '{self._escape_string(name)}';"
                )
        
        cypher_lines.append("")
        
        # Generate relationship creation statements
        cypher_lines.append("// Create Relationships")
        for edge in edges:
            source = edge['source']
            target = edge['target']
            rel_type = self._sanitize_rel_type(edge.get('type', 'RELATED_TO'))
            
            cypher_lines.append(
                f"MATCH (a), (b) WHERE a.id = '{source}' AND b.id = '{target}' "
                f"MERGE (a)-[r:{rel_type}]->(b);"
            )
        
        cypher_script = '\n'.join(cypher_lines)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(cypher_script)
            print(f"Cypher script saved to: {output_path}")
        
        return cypher_script
    
    def _escape_string(self, s: str) -> str:
        """Escape special characters in strings for Cypher"""
        if not isinstance(s, str):
            s = str(s)
        return s.replace("'", "\\'").replace("\\", "\\\\")
    
    def clear_database(self):
        """Clear all data from the database (use with caution!)"""
        if not self._driver:
            if not self.connect():
                return
        
        confirm = input("WARNING: This will delete ALL data. Type 'DELETE' to confirm: ")
        if confirm != 'DELETE':
            print("Operation cancelled")
            return
        
        queries = [
            "MATCH (n) DETACH DELETE n",
            "SHOW CONSTRAINTS YIELD name CALL dbms.dropConstraint(name) RETURN *",
            "SHOW INDEXES YIELD name CALL db.index.drop(name) RETURN *"
        ]
        
        with self._driver.session() as session:
            for query in queries:
                try:
                    session.run(query).consume()
                except Exception as e:
                    print(f"Warning: {e}")
        
        print("Database cleared")
    
    def query(self, cypher_query: str) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query
        
        Args:
            cypher_query: Cypher query to execute
            
        Returns:
            List of result records
        """
        if not self._driver:
            if not self.connect():
                return []
        
        try:
            with self._driver.session() as session:
                result = session.run(cypher_query)
                return [record.data() for record in result]
        except Exception as e:
            print(f"Query error: {e}")
            return []
    
    def get_schema(self) -> Dict[str, Any]:
        """Get database schema information"""
        if not self._driver:
            if not self.connect():
                return {}
        
        schema = {
            'labels': [],
            'relationship_types': [],
            'property_keys': []
        }
        
        with self._driver.session() as session:
            # Get node labels
            result = session.run("CALL db.labels()")
            schema['labels'] = [record['label'] for record in result]
            
            # Get relationship types
            result = session.run("CALL db.relationshipTypes()")
            schema['relationship_types'] = [record['relationshipType'] for record in result]
            
            # Get property keys
            result = session.run("CALL db.propertyKeys()")
            schema['property_keys'] = [record['propertyKey'] for record in result]
        
        return schema
