"""
Intent Classifier - PyTorch-based model for routing user queries to appropriate agents
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel


class IntentClassifierModel(nn.Module):
    """Lightweight PyTorch model for intent classification"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, num_classes: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class IntentClassifier:
    """Fine-tuned intent classifier for agent routing"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained sentence transformer for embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Initialize classifier
        self.model = IntentClassifierModel()
        self.model.to(self.device)
        
        # Load fine-tuned weights if available
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✓ Loaded fine-tuned intent classifier from {model_path}")
        else:
            print("⚠ Using base intent classifier (not fine-tuned)")
        
        self.model.eval()
        
        # Intent labels
        self.labels = ["sales", "research", "general"]
        
        # Keyword-based fallback for better accuracy
        self.sales_keywords = [
            "buy", "purchase", "price", "pricing", "cost", "demo", "trial",
            "subscribe", "plan", "upgrade", "discount", "payment", "invoice"
        ]
        self.research_keywords = [
            "research", "data", "analysis", "insight", "study", "report",
            "statistics", "trends", "survey", "feedback", "users", "behavior"
        ]
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get sentence embedding using transformer model"""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True
            ).to(self.device)
            
            outputs = self.encoder(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings
    
    async def classify(self, text: str) -> str:
        """Classify user intent"""
        # Get embedding
        embedding = self._get_embedding(text)
        
        # Get model prediction
        with torch.no_grad():
            logits = self.model(embedding)
            probabilities = torch.softmax(logits, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item()
        
        predicted_intent = self.labels[predicted_idx]
        
        # Use keyword-based fallback for low confidence predictions
        if confidence < 0.6:
            predicted_intent = self._keyword_classify(text)
        
        print(f"🎯 Intent: {predicted_intent} (confidence: {confidence:.2f})")
        return predicted_intent
    
    def _keyword_classify(self, text: str) -> str:
        """Fallback keyword-based classification"""
        text_lower = text.lower()
        
        sales_score = sum(1 for keyword in self.sales_keywords if keyword in text_lower)
        research_score = sum(1 for keyword in self.research_keywords if keyword in text_lower)
        
        if sales_score > research_score and sales_score > 0:
            return "sales"
        elif research_score > sales_score and research_score > 0:
            return "research"
        else:
            return "general"
    
    def train(self, training_data: List[Dict], epochs: int = 10, lr: float = 0.001):
        """Fine-tune the intent classifier"""
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for item in training_data:
                text = item["text"]
                label_idx = self.labels.index(item["label"])
                
                # Get embedding
                embedding = self._get_embedding(text)
                
                # Forward pass
                optimizer.zero_grad()
                logits = self.model(embedding)
                
                # Calculate loss
                target = torch.tensor([label_idx]).to(self.device)
                loss = criterion(logits, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                predicted = torch.argmax(logits, dim=1).item()
                correct += (predicted == label_idx)
                total += 1
            
            accuracy = correct / total
            avg_loss = total_loss / total
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        self.model.eval()
        print("✓ Training complete")
    
    def save_model(self, path: str):
        """Save fine-tuned model"""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")


# Sample training data for fine-tuning
SAMPLE_TRAINING_DATA = [
    # Sales examples
    {"text": "How much does your product cost?", "label": "sales"},
    {"text": "I'd like to schedule a demo", "label": "sales"},
    {"text": "What's included in the pro plan?", "label": "sales"},
    {"text": "Can I get a discount for annual billing?", "label": "sales"},
    {"text": "I want to upgrade my subscription", "label": "sales"},
    
    # Research examples
    {"text": "What do users think about feature X?", "label": "research"},
    {"text": "Show me the latest survey results", "label": "research"},
    {"text": "What are the main pain points for our users?", "label": "research"},
    {"text": "Can you analyze the user feedback from last month?", "label": "research"},
    {"text": "What trends are you seeing in user behavior?", "label": "research"},
    
    # General examples
    {"text": "Hello, how are you?", "label": "general"},
    {"text": "What can you help me with?", "label": "general"},
    {"text": "Tell me about your capabilities", "label": "general"},
]
