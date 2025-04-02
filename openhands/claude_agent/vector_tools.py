"""
Vector database tools for the Claude Agent.

This module provides vector database tools for the Claude Agent,
including document loading, storage, and retrieval.
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger("claude_agent")

class VectorDBManager:
    """Vector database manager."""
    
    def __init__(self, db_path: str = "app/db"):
        """
        Initialize the vector database manager.
        
        Args:
            db_path: Path to the database directory
        """
        self.db_path = db_path
        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)
        # Initialize embedding model
        try:
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings()
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            self.embeddings = None
            
        # Database cache
        self.db_cache = {}
    
    def _get_db_path(self, db_name: str) -> str:
        """
        Get database storage path.
        
        Args:
            db_name: Name of the database
            
        Returns:
            Path to the database
        """
        return os.path.join(self.db_path, db_name)
    
    def create_or_load_db(self, db_name: str, documents: Optional[List[Any]] = None) -> Any:
        """
        Create or load vector database.
        
        Args:
            db_name: Name of the database
            documents: Optional list of documents
            
        Returns:
            Database instance or error message
        """
        db_path = self._get_db_path(db_name)
        
        # If already in cache, return directly
        if db_name in self.db_cache:
            return self.db_cache[db_name]
            
        try:
            # Check if database already exists
            from langchain_community.vectorstores import Chroma
            
            if os.path.exists(db_path) and os.path.isdir(db_path):
                # Load existing database
                db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
            else:
                # If documents provided, create new database
                if documents:
                    db = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory=db_path
                    )
                    db.persist()
                else:
                    # Create empty database
                    db = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
                    db.persist()
                    
            # Cache database instance
            self.db_cache[db_name] = db
            return db
        except Exception as e:
            return f"Error creating or loading database: {str(e)}"
    
    def add_documents(self, db_name: str, documents: List[Any]) -> str:
        """
        Add documents to database.
        
        Args:
            db_name: Name of the database
            documents: List of documents
            
        Returns:
            Success or error message
        """
        try:
            db = self.create_or_load_db(db_name)
            if isinstance(db, str):  # Error message
                return db
                
            # Add documents
            db.add_documents(documents)
            db.persist()
            return f"Successfully added {len(documents)} documents to database {db_name}"
        except Exception as e:
            return f"Error adding documents: {str(e)}"
    
    def search(self, db_name: str, query: str, k: int = 5) -> Union[List[Dict[str, Any]], str]:
        """
        Search for similar content in database.
        
        Args:
            db_name: Name of the database
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results or error message
        """
        try:
            db = self.create_or_load_db(db_name)
            if isinstance(db, str):  # Error message
                return db
                
            results = db.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
                
            return formatted_results
        except Exception as e:
            return f"Error searching: {str(e)}"
    
    def load_and_split_document(self, file_path: str) -> Union[List[Any], str]:
        """
        Load and split document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks or error message
        """
        try:
            # Choose appropriate loader based on file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
            else:
                # Default to text file
                loader = TextLoader(file_path, encoding='utf-8')
                
            documents = loader.load()
            
            # Text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Add source file information to each document
            for doc in documents:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path
                    
            # Split documents
            split_docs = text_splitter.split_documents(documents)
            
            return split_docs
        except Exception as e:
            return f"Error loading document: {str(e)}"


def get_vector_tools() -> List[Any]:
    """
    Get vector database tools.
    
    Returns:
        List of vector database tools
    """
    try:
        from langchain.tools import Tool
        
        # Initialize vector database manager
        vector_db_manager = VectorDBManager()
        
        tools = []
        
        # Create tool for adding file to vector database
        def add_file_to_vectordb(file_path: str, db_name: str = "default") -> str:
            """
            Add file to vector database.
            
            Args:
                file_path: Path to the file
                db_name: Name of the database (default: "default")
                
            Returns:
                Success or error message
            """
            try:
                # Load and split document
                documents = vector_db_manager.load_and_split_document(file_path)
                if isinstance(documents, str):  # Error message
                    return documents
                    
                # Add to database
                result = vector_db_manager.add_documents(db_name, documents)
                return result
            except Exception as e:
                return f"Error adding file to vector database: {str(e)}"
        
        add_file_tool = Tool(
            name="AddFileToVectorDB",
            func=add_file_to_vectordb,
            description="Add file to vector database. Input file path and optional database name."
        )
        tools.append(add_file_tool)
        
        # Create vector database search tool
        def search_vectordb(query: str, db_name: str = "default", k: int = 5) -> str:
            """
            Search vector database.
            
            Args:
                query: Search query
                db_name: Name of the database (default: "default")
                k: Number of results to return (default: 5)
                
            Returns:
                Search results or error message
            """
            try:
                results = vector_db_manager.search(db_name, query, k)
                if isinstance(results, str):  # Error message
                    return results
                    
                # Format results as readable text
                output = f"Found {len(results)} results in database {db_name}:\n\n"
                
                for i, item in enumerate(results):
                    output += f"Result {i+1} (similarity: {1.0 - item['score']:.4f}):\n"
                    output += f"Content: {item['content'][:200]}...\n"
                    output += f"Source: {item['metadata'].get('source', 'unknown')}\n\n"
                    
                return output
            except Exception as e:
                return f"Error searching vector database: {str(e)}"
        
        search_tool = Tool(
            name="SearchVectorDB",
            func=search_vectordb,
            description="Search for similar content in vector database. Input search query, database name, and number of results."
        )
        tools.append(search_tool)
        
        return tools
    
    except ImportError:
        logger.warning("Could not create vector tools: required modules not found")
        return []
    except Exception as e:
        logger.error(f"Error creating vector tools: {e}")
        return []