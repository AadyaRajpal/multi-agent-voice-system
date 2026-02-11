"""
RAG Knowledge Base - Retrieval-Augmented Generation system
Connects agents with task-specific knowledge bases
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    """RAG system for context-aware agent responses"""
    
    def __init__(self, persist_directory: str = "./data/chroma"):
        """Initialize the knowledge base with vector storage"""
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for vector storage
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create collections for different agent types
        self.sales_collection = self._get_or_create_collection("sales_knowledge")
        self.research_collection = self._get_or_create_collection("research_knowledge")
        
        print("✓ Knowledge base initialized")
    
    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.client.get_collection(name)
        except:
            return self.client.create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"}
            )
    
    def add_documents(
        self,
        documents: List[str],
        metadata: List[Dict],
        agent_type: str = "sales"
    ):
        """Add documents to the knowledge base"""
        
        collection = (
            self.sales_collection if agent_type == "sales"
            else self.research_collection
        )
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(documents).tolist()
        
        # Generate IDs
        ids = [f"{agent_type}_{i}" for i in range(len(documents))]
        
        # Add to collection
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        
        print(f"✓ Added {len(documents)} documents to {agent_type} knowledge base")
    
    async def query(
        self,
        query: str,
        agent_type: str = "sales",
        top_k: int = 3
    ) -> str:
        """Query the knowledge base for relevant context"""
        
        collection = (
            self.sales_collection if agent_type == "sales"
            else self.research_collection
        )
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query the collection
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        # Format results
        if results['documents'] and len(results['documents'][0]) > 0:
            context = "\n\n".join(results['documents'][0])
            return context
        else:
            return ""
    
    def load_from_files(self, directory: str, agent_type: str = "sales"):
        """Load knowledge base from text files"""
        
        data_path = Path(directory)
        if not data_path.exists():
            print(f"⚠ Directory {directory} not found")
            return
        
        documents = []
        metadata = []
        
        for file_path in data_path.glob("*.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Split into chunks (simple chunking)
                chunks = self._chunk_text(content, chunk_size=500)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        "source": file_path.name,
                        "chunk_id": i,
                        "agent_type": agent_type
                    })
        
        if documents:
            self.add_documents(documents, metadata, agent_type)
    
    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def initialize_with_sample_data(self):
        """Initialize knowledge base with sample data"""
        
        # Sales knowledge
        sales_docs = [
            """
            Our product comes in three tiers: Basic ($49/month), Pro ($99/month), 
            and Enterprise (custom pricing). The Basic plan includes core features 
            and email support. Pro adds priority support and advanced analytics. 
            Enterprise includes everything plus dedicated support and custom integrations.
            """,
            """
            We offer a 14-day free trial with no credit card required. You can 
            upgrade or downgrade at any time. Annual billing gives you 2 months free. 
            We accept all major credit cards and can set up invoicing for enterprise customers.
            """,
            """
            Our demo showcases the key features: real-time collaboration, advanced 
            reporting, custom workflows, and integrations with 100+ tools. Demos 
            typically last 30 minutes and can be scheduled at your convenience.
            """
        ]
        
        sales_metadata = [
            {"source": "pricing", "category": "pricing"},
            {"source": "billing", "category": "pricing"},
            {"source": "demo", "category": "demo"}
        ]
        
        self.add_documents(sales_docs, sales_metadata, "sales")
        
        # Research knowledge
        research_docs = [
            """
            Recent user research shows that 78% of users prioritize ease of use 
            over advanced features. The top three pain points identified are: 
            complex onboarding (mentioned by 45% of users), lack of mobile support 
            (38%), and limited integration options (32%).
            """,
            """
            User behavior analysis reveals that power users (top 10%) generate 60% 
            of total engagement. These users typically access the platform daily 
            and use an average of 8 different features. Retention is highest among 
            users who complete onboarding within the first 24 hours.
            """,
            """
            Latest survey data (n=1,200 respondents) indicates 85% satisfaction rate. 
            Key strengths: reliability (92% positive), customer support (88% positive), 
            and feature set (81% positive). Areas for improvement: pricing clarity 
            (62% positive) and mobile experience (58% positive).
            """
        ]
        
        research_metadata = [
            {"source": "user_research", "category": "insights"},
            {"source": "behavior_analysis", "category": "analytics"},
            {"source": "satisfaction_survey", "category": "feedback"}
        ]
        
        self.add_documents(research_docs, research_metadata, "research")
        
        print("✓ Sample knowledge base initialized")


# Initialize knowledge base with sample data on first import
def setup_knowledge_base():
    """Setup function to initialize knowledge base"""
    kb = KnowledgeBase()
    kb.initialize_with_sample_data()
    return kb
