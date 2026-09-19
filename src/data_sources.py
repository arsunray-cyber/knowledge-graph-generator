"""
Data Source Loader Module
Supports: websites, databases, documents, and text chunks
"""

import os
from typing import List, Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
import hashlib


class DataSourceLoader:
    """Load data from various sources"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.loaded_data = []
        
    def load(self, source_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load data from a source
        
        Args:
            source_type: Type of data source ('website', 'database', 'document', 'text')
            **kwargs: Source-specific parameters
            
        Returns:
            List of dictionaries with 'content' and 'metadata'
        """
        if source_type == 'website':
            return self._load_website(**kwargs)
        elif source_type == 'database':
            return self._load_database(**kwargs)
        elif source_type == 'document':
            return self._load_document(**kwargs)
        elif source_type == 'text':
            return self._load_text(**kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def _load_website(self, url: str, max_depth: int = 1, 
                     follow_links: bool = False) -> List[Dict[str, Any]]:
        """Load content from a website"""
        results = []
        visited_urls = set()
        
        def scrape_url(current_url: str, depth: int):
            if depth > max_depth or current_url in visited_urls:
                return
                
            visited_urls.add(current_url)
            
            try:
                headers = {
                    'User-Agent': self.config.get('website', {}).get(
                        'user_agent', 
                        'Mozilla/5.0 (compatible; KnowledgeGraphBot/1.0)'
                    )
                }
                response = requests.get(current_url, headers=headers, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'lxml')
                
                # Extract main content
                for tag in soup(['script', 'style', 'nav', 'footer']):
                    tag.decompose()
                
                content = soup.get_text(separator='\n', strip=True)
                
                if content:
                    results.append({
                        'content': content,
                        'metadata': {
                            'source': current_url,
                            'source_type': 'website',
                            'url': current_url,
                            'title': soup.title.string if soup.title else '',
                        }
                    })
                
                # Follow links if requested
                if follow_links and depth < max_depth:
                    base_domain = '/'.join(current_url.split('/')[:3])
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if href.startswith(base_domain) or href.startswith('/'):
                            next_url = href if href.startswith('http') else f"{base_domain}{href}"
                            scrape_url(next_url, depth + 1)
                            
            except Exception as e:
                print(f"Error scraping {current_url}: {str(e)}")
        
        scrape_url(url, 0)
        self.loaded_data.extend(results)
        return results
    
    def _load_database(self, connection_string: str, query: str, 
                      **db_params) -> List[Dict[str, Any]]:
        """Load data from a database"""
        # This is a placeholder - implement based on your database type
        # Example implementations for PostgreSQL, MySQL, etc.
        results = []
        
        db_type = db_params.get('type', 'postgresql')
        
        if db_type == 'postgresql':
            try:
                import psycopg2
                conn = psycopg2.connect(
                    host=db_params.get('host', 'localhost'),
                    port=db_params.get('port', 5432),
                    database=db_params.get('database'),
                    user=db_params.get('user'),
                    password=db_params.get('password')
                )
                cursor = conn.cursor()
                cursor.execute(query)
                
                columns = [desc[0] for desc in cursor.description]
                for row in cursor.fetchall():
                    content = ' | '.join([f"{col}: {val}" for col, val in zip(columns, row)])
                    results.append({
                        'content': content,
                        'metadata': {
                            'source': 'database',
                            'source_type': 'database',
                            'table': query.split('FROM')[1].split()[0] if 'FROM' in query else 'unknown',
                        }
                    })
                
                cursor.close()
                conn.close()
            except ImportError:
                raise ImportError("Install psycopg2 for PostgreSQL support: pip install psycopg2-binary")
                
        elif db_type == 'sqlite':
            import sqlite3
            db_path = db_params.get('path', 'database.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                content = ' | '.join([f"{col}: {val}" for col, val in zip(columns, row)])
                results.append({
                    'content': content,
                    'metadata': {
                        'source': 'database',
                        'source_type': 'database',
                    }
                })
            
            cursor.close()
            conn.close()
        
        self.loaded_data.extend(results)
        return results
    
    def _load_document(self, path: str) -> List[Dict[str, Any]]:
        """Load content from a document file"""
        results = []
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document not found: {path}")
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.pdf':
            content = self._read_pdf(path)
        elif ext == '.docx':
            content = self._read_docx(path)
        elif ext in ['.txt', '.md']:
            content = self._read_text(path)
        else:
            raise ValueError(f"Unsupported document format: {ext}")
        
        if content:
            results.append({
                'content': content,
                'metadata': {
                    'source': path,
                    'source_type': 'document',
                    'filename': os.path.basename(path),
                    'file_type': ext,
                }
            })
        
        self.loaded_data.extend(results)
        return results
    
    def _read_pdf(self, path: str) -> str:
        """Read PDF file"""
        content_parts = []
        with open(path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    content_parts.append(f"[Page {page_num + 1}]\n{text}")
        return '\n\n'.join(content_parts)
    
    def _read_docx(self, path: str) -> str:
        """Read DOCX file"""
        doc = docx.Document(path)
        content_parts = []
        for para in doc.paragraphs:
            if para.text.strip():
                content_parts.append(para.text)
        return '\n\n'.join(content_parts)
    
    def _read_text(self, path: str) -> str:
        """Read text or markdown file"""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Load from raw text string"""
        results = [{
            'content': text,
            'metadata': metadata or {
                'source': 'text_input',
                'source_type': 'text',
            }
        }]
        self.loaded_data.extend(results)
        return results
    
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Get all loaded data"""
        return self.loaded_data
    
    def clear(self):
        """Clear all loaded data"""
        self.loaded_data = []
