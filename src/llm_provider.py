"""
LLM Provider Module
Supports: Grok, Ollama, OpenRouter
"""

import os
import json
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class LLMProvider:
    """Interface for different LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get('provider', 'ollama')
        self.model = config.get('model', 'llama2')
        self.temperature = config.get('temperature', 0.7)
        self.max_tokens = config.get('max_tokens', 2000)
        
        # Initialize provider-specific settings
        if self.provider == 'grok':
            self.api_key = config.get('grok_api_key') or os.getenv('GROK_API_KEY')
            self.base_url = "https://api.x.ai/v1"
        elif self.provider == 'openrouter':
            self.api_key = config.get('openrouter_api_key') or os.getenv('OPENROUTER_API_KEY')
            self.base_url = "https://openrouter.ai/api/v1"
        elif self.provider == 'ollama':
            self.base_url = config.get('ollama_base_url', 'http://localhost:11434')
            self.api_key = None  # Ollama doesn't need API key
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        """
        Extract entities (nodes) and relationships (edges) from text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with 'nodes' and 'relationships' lists
        """
        prompt = self._create_extraction_prompt(text)
        response = self._call_llm(prompt)
        
        return self._parse_response(response)
    
    def _create_extraction_prompt(self, text: str) -> str:
        """Create prompt for entity and relationship extraction"""
        return f"""You are a knowledge graph extraction expert. Analyze the following text and extract:
1. Entities (nodes) - important concepts, people, places, organizations, things
2. Relationships (edges) - how entities are connected

For each entity, provide:
- id: unique identifier (camelCase)
- label: entity type (e.g., Person, Organization, Location, Concept)
- properties: dict with name, description, and other relevant attributes

For each relationship, provide:
- source: id of the source entity
- target: id of the target entity  
- type: relationship type (e.g., WORKS_FOR, LOCATED_IN, RELATED_TO)
- properties: dict with any additional information

Return ONLY valid JSON in this exact format:
{{
    "nodes": [
        {{"id": "entityId", "label": "EntityType", "properties": {{"name": "Entity Name", "description": "Brief description"}}}}
    ],
    "relationships": [
        {{"source": "entityId1", "target": "entityId2", "type": "RELATIONSHIP_TYPE", "properties": {{"description": "Relationship description"}}}}
    ]
}}

Text to analyze:
{text}

JSON Output:"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt"""
        if self.provider == 'grok':
            return self._call_grok(prompt)
        elif self.provider == 'openrouter':
            return self._call_openrouter(prompt)
        elif self.provider == 'ollama':
            return self._call_ollama(prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_grok(self, prompt: str) -> str:
        """Call Grok API"""
        import requests
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling Grok: {e}")
            return '{"nodes": [], "relationships": []}'
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API"""
        import requests
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost',  # Required by OpenRouter
            'X-Title': 'Knowledge Graph Generator'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/chat/completions',
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error calling OpenRouter: {e}")
            return '{"nodes": [], "relationships": []}'
    
    def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API (local)"""
        import requests
        
        payload = {
            'model': self.model,
            'prompt': prompt,
            'stream': False,
            'options': {
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return '{"nodes": [], "relationships": []}'
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        # Try to extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response)
        
        if json_match:
            try:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validate structure
                if 'nodes' not in result:
                    result['nodes'] = []
                if 'relationships' not in result:
                    result['relationships'] = []
                
                return result
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                print(f"Response was: {response[:500]}...")
        
        # Fallback: return empty structure
        return {'nodes': [], 'relationships': []}
    
    def batch_extract(self, texts: List[str], batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Extract entities and relationships from multiple texts
        
        Args:
            texts: List of texts to process
            batch_size: Number of texts to process in parallel (not implemented, sequential for now)
            
        Returns:
            List of extraction results
        """
        results = []
        for i, text in enumerate(texts):
            print(f"Processing text {i+1}/{len(texts)}...")
            result = self.extract_entities_and_relationships(text)
            results.append(result)
        return results
    
    def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text"""
        prompt = f"""Summarize the following text in {max_length} words or less. 
Focus on the main entities and their relationships.

Text:
{text}

Summary:"""
        
        return self._call_llm(prompt)
