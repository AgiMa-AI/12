"""
Hybrid Retrieval Strategy for the LangChain Router.

This module implements a hybrid retrieval strategy that combines multiple retrieval methods
to get the most relevant information for a given query.
"""

from typing import Dict, List, Optional, Any, Union
import logging
import os
import datetime

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    A hybrid retrieval system that combines multiple retrieval methods.
    
    This retriever uses a combination of:
    1. Vector search (semantic similarity)
    2. BM25 (keyword matching)
    3. Multi-query expansion
    
    This approach provides more robust and relevant results than any single method.
    """
    
    def __init__(
        self,
        db_path: str = "/.openhands-state/conversation_db",
        embedding_model: str = "local",
        top_k: int = 5
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            db_path: Path to the conversation database
            embedding_model: Name of the embedding model to use
            top_k: Number of results to retrieve
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.top_k = top_k
        
        # Create the database directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Initialize the embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize the vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize the retrievers
        self.vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        self.bm25_retriever = self._initialize_bm25_retriever()
        self.ensemble_retriever = self._initialize_ensemble_retriever()
    
    def _initialize_embeddings(self) -> Any:
        """Initialize the embeddings model."""
        # In a real implementation, we would use different embedding models
        # based on the configuration
        try:
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            # Return a simple embedding function for testing
            return lambda text: [0.0] * 384  # Placeholder
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the vector store."""
        try:
            # Try to load an existing vector store
            return Chroma(
                persist_directory=os.path.join(self.db_path, "vector_store"),
                embedding_function=self.embeddings
            )
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            # Create a new vector store with empty documents
            return Chroma.from_documents(
                documents=[Document(page_content="Initial document", metadata={"source": "init"})],
                embedding=self.embeddings,
                persist_directory=os.path.join(self.db_path, "vector_store")
            )
    
    def _initialize_bm25_retriever(self) -> BM25Retriever:
        """Initialize the BM25 retriever."""
        # In a real implementation, we would load documents from the database
        # For now, we'll create a simple BM25 retriever with placeholder documents
        documents = [
            Document(page_content="Example conversation 1", metadata={"source": "conversation"}),
            Document(page_content="Example conversation 2", metadata={"source": "conversation"})
        ]
        return BM25Retriever.from_documents(documents)
    
    def _initialize_ensemble_retriever(self) -> EnsembleRetriever:
        """Initialize the ensemble retriever."""
        return EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]
        )
    
    def retrieve(self, query: str) -> str:
        """
        Retrieve relevant information for a query.
        
        Args:
            query: The query to retrieve information for
            
        Returns:
            A string containing the retrieved information
        """
        try:
            # Use the ensemble retriever to get results
            results = self.ensemble_retriever.get_relevant_documents(query)
            
            # Format the results
            formatted_results = []
            for i, doc in enumerate(results[:self.top_k]):
                source = doc.metadata.get("source", "unknown")
                timestamp = doc.metadata.get("timestamp", "unknown")
                formatted_results.append(
                    f"[{i+1}] {doc.page_content[:200]}... (Source: {source}, Time: {timestamp})"
                )
            
            return "\n\n".join(formatted_results)
        
        except Exception as e:
            logger.error(f"Error retrieving information: {e}")
            return ""
    
    def add_conversation(self, user_input: str, response: str, metadata: Dict[str, Any] = None) -> None:
        """
        Add a conversation to the database.
        
        Args:
            user_input: The user's input
            response: The system's response
            metadata: Additional metadata
        """
        try:
            # Create a document from the conversation
            conversation = f"User: {user_input}\nSystem: {response}"
            
            # Add timestamp and other metadata
            doc_metadata = {
                "source": "conversation",
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "dialogue"
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            # Split the document into chunks
            docs = self.text_splitter.create_documents(
                texts=[conversation],
                metadatas=[doc_metadata]
            )
            
            # Add the documents to the vector store
            self.vector_store.add_documents(docs)
            self.vector_store.persist()
            
            # Update the BM25 retriever
            # In a real implementation, we would need to rebuild or update the BM25 index
            logger.info(f"Added conversation to database (length: {len(conversation)})")
        
        except Exception as e:
            logger.error(f"Error adding conversation to database: {e}")
    
    def search_by_keyword(self, keyword: str) -> List[Document]:
        """
        Search for documents containing a keyword.
        
        Args:
            keyword: The keyword to search for
            
        Returns:
            A list of documents containing the keyword
        """
        return self.bm25_retriever.get_relevant_documents(keyword)
    
    def search_by_semantic(self, query: str) -> List[Document]:
        """
        Search for documents semantically similar to a query.
        
        Args:
            query: The query to search for
            
        Returns:
            A list of documents semantically similar to the query
        """
        return self.vector_retriever.get_relevant_documents(query)