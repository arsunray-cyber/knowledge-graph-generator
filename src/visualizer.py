"""
Graph Visualizer Module
Creates interactive visualizations of knowledge graphs
"""

import networkx as nx
from typing import Dict, Any, Optional, List
import html


class GraphVisualizer:
    """Visualize knowledge graphs using pyvis and other tools"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_file = self.config.get('output_file', 'knowledge_graph.html')
        self.notebook_mode = self.config.get('notebook_mode', False)
        self.height = self.config.get('height', '750px')
        self.width = self.config.get('width', '100%')
        self.bgcolor = self.config.get('bgcolor', '#ffffff')
        self.font_color = self.config.get('font_color', '#000000')
        
    def visualize(self, graph_builder, output_path: Optional[str] = None) -> str:
        """
        Create interactive visualization of the graph
        
        Args:
            graph_builder: GraphBuilder instance with graph data
            output_path: Optional custom output path
            
        Returns:
            Path to the generated HTML file
        """
        from pyvis.network import Network
        
        output_path = output_path or self.output_file
        
        # Create pyvis network
        net = Network(
            height=self.height,
            width=self.width,
            bgcolor=self.bgcolor,
            font_color=self.font_color,
            notebook=self.notebook_mode,
            directed=True
        )
        
        # Configure physics for better layout
        net.set_options("""
        {
            "physics": {
                "enabled": true,
                "barnesHut": {
                    "gravitationalConstant": -2000,
                    "centralGravity": 0.3,
                    "springLength": 95,
                    "springConstant": 0.04,
                    "damping": 0.09
                },
                "stabilization": {
                    "enabled": true,
                    "iterations": 100
                }
            },
            "interaction": {
                "hover": true,
                "tooltipDelay": 200,
                "hideEdgesOnDrag": false
            }
        }
        """)
        
        # Add nodes with colors based on their labels
        label_colors = {}
        color_palette = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
            '#F7DC6F', '#BB8FCE', '#82E0AA', '#F1948A', '#85C1E9'
        ]
        
        nodes = graph_builder.get_nodes()
        for i, node in enumerate(nodes):
            label = node.get('label', 'Entity')
            
            # Assign color based on label type
            if label not in label_colors:
                label_colors[label] = color_palette[len(label_colors) % len(color_palette)]
            
            # Create tooltip with all properties
            tooltip = self._create_node_tooltip(node)
            
            net.add_node(
                node['id'],
                label=node.get('properties', {}).get('name', node['id']),
                title=tooltip,
                color=label_colors[label],
                shape='dot',
                size=15 + min(node.get('count', 1) * 2, 20)  # Size based on frequency
            )
        
        # Add edges with different styles based on relationship type
        edges = graph_builder.get_edges()
        edge_styles = {}
        
        for edge in edges:
            rel_type = edge.get('type', 'RELATED_TO')
            
            if rel_type not in edge_styles:
                edge_styles[rel_type] = {
                    'color': '#666666',
                    'dashes': False
                }
            
            tooltip = f"{rel_type}"
            if edge.get('properties', {}).get('description'):
                tooltip += f": {edge['properties']['description']}"
            
            net.add_edge(
                edge['source'],
                edge['target'],
                title=tooltip,
                label=rel_type,
                color=edge_styles[rel_type]['color'],
                dashes=edge_styles[rel_type]['dashes'],
                arrows='to',
                smooth={'type': 'curvedCW', 'roundness': 0.2}
            )
        
        # Add legend as nodes in a separate section
        self._add_legend(net, label_colors)
        
        # Generate HTML
        net.save_graph(output_path)
        print(f"Graph visualization saved to: {output_path}")
        
        return output_path
    
    def _create_node_tooltip(self, node: Dict[str, Any]) -> str:
        """Create HTML tooltip for a node"""
        props = node.get('properties', {})
        
        tooltip_parts = [
            f"<div style='padding: 10px; max-width: 300px;'>",
            f"<h4 style='margin: 0 0 10px 0;'>{props.get('name', node['id'])}</h4>",
            f"<p><strong>Type:</strong> {node.get('label', 'Entity')}</p>"
        ]
        
        if props.get('description'):
            desc = props['description'][:200] + ('...' if len(props['description']) > 200 else '')
            tooltip_parts.append(f"<p><strong>Description:</strong> {desc}</p>")
        
        # Add other properties
        for key, value in props.items():
            if key not in ['name', 'description']:
                tooltip_parts.append(f"<p><strong>{key}:</strong> {value}</p>")
        
        if node.get('count', 1) > 1:
            tooltip_parts.append(f"<p><strong>Occurrences:</strong> {node['count']}</p>")
        
        tooltip_parts.append("</div>")
        
        return ''.join(tooltip_parts)
    
    def _add_legend(self, net, label_colors: Dict[str, str]) -> None:
        """Add legend to the visualization"""
        # Legend is typically added via HTML manipulation
        # For pyvis, we can add it as hidden nodes or modify the HTML after generation
        pass
    
    def visualize_with_plotly(self, graph_builder, 
                             output_path: Optional[str] = None) -> str:
        """
        Alternative visualization using Plotly
        
        Args:
            graph_builder: GraphBuilder instance
            output_path: Output HTML file path
            
        Returns:
            Path to generated HTML file
        """
        import plotly.graph_objects as go
        
        output_path = output_path or self.output_file.replace('.html', '_plotly.html')
        
        nodes = graph_builder.get_nodes()
        edges = graph_builder.get_edges()
        
        # Prepare node positions using spring layout
        G = graph_builder.graph
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge traces
        edge_x, edge_y = [], []
        for edge in edges:
            source = edge['source']
            target = edge['target']
            
            if source in pos and target in pos:
                x0, y0 = pos[source]
                x1, y1 = pos[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Create node trace
        node_x, node_y = [], []
        node_colors = []
        node_labels = []
        node_sizes = []
        
        label_to_color = {}
        color_idx = 0
        
        for node_id in pos:
            x, y = pos[node_id]
            node_x.append(x)
            node_y.append(y)
            
            node_data = next((n for n in nodes if n['id'] == node_id), {})
            label = node_data.get('label', 'Entity')
            
            if label not in label_to_color:
                label_to_color[label] = color_idx
                color_idx += 1
            
            node_colors.append(label_to_color[label])
            node_labels.append(node_data.get('properties', {}).get('name', node_id))
            node_sizes.append(10 + min(node_data.get('count', 1) * 2, 15))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                line=dict(width=1, color='black'),
                showscale=True,
                colorbar=dict(title='Node Types')
            ),
            text=node_labels,
            textposition="top center",
            hoverinfo='text'
        )
        
        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
        )
        
        fig.update_layout(
            title=f"Knowledge Graph ({len(nodes)} nodes, {len(edges)} relationships)",
            height=750,
            width=1200
        )
        
        fig.write_html(output_path)
        print(f"Plotly visualization saved to: {output_path}")
        
        return output_path
    
    def generate_stats_report(self, graph_builder) -> str:
        """Generate a text report with graph statistics"""
        stats = graph_builder.get_graph_stats()
        central_nodes = graph_builder.get_central_nodes(top_k=5)
        components = graph_builder.find_connected_components()
        
        report = [
            "=" * 60,
            "KNOWLEDGE GRAPH STATISTICS REPORT",
            "=" * 60,
            "",
            "GRAPH OVERVIEW:",
            f"  - Total Nodes: {stats['num_nodes']}",
            f"  - Total Edges: {stats['num_edges']}",
            f"  - Node Types: {stats['node_types']}",
            f"  - Relationship Types: {stats['relationship_types']}",
            f"  - Graph Density: {stats['density']:.4f}",
            "",
            "CONNECTED COMPONENTS:",
            f"  - Number of Components: {len(components)}",
            f"  - Largest Component Size: {max(len(c) for c in components) if components else 0}",
            "",
            "TOP 5 CENTRAL NODES:",
        ]
        
        for i, node in enumerate(central_nodes, 1):
            name = node.get('properties', {}).get('name', node['id'])
            report.append(f"  {i}. {name} (Type: {node['label']}, Centrality: {node['centrality_score']:.3f})")
        
        report.extend([
            "",
            "=" * 60,
        ])
        
        return '\n'.join(report)
