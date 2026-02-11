"""
Sales Agent - Specialized for handling sales inquiries, pricing, and demos
"""

import json
from typing import List, Dict, Optional
from cerebras.cloud.sdk import Cerebras


class SalesAgent:
    """Handles sales-related conversations with product knowledge"""
    
    def __init__(self, api_key: str):
        self.client = Cerebras(api_key=api_key)
        self.persona = """
You are a professional sales consultant with deep product knowledge.
You help customers understand product value, pricing, and features.
You're persuasive but not pushy, always focusing on customer needs.
Keep responses concise and action-oriented for voice delivery (2-3 sentences max).
"""
    
    async def process(
        self,
        message: str,
        context: Optional[str] = None,
        history: List[Dict] = None
    ) -> str:
        """Process a sales-related message"""
        
        # Build conversation context
        messages = [{"role": "system", "content": self.persona}]
        
        # Add conversation history
        if history:
            messages.extend(history[-6:])  # Last 3 exchanges
        
        # Add RAG context if available
        if context:
            context_prompt = f"""
Relevant product information:
{context}

Use this information to provide accurate, helpful responses.
"""
            messages.append({"role": "system", "content": context_prompt})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = self.client.chat.completions.create(
            model="llama3.1-8b",
            messages=messages,
            temperature=0.7,
            max_tokens=150  # Keep responses short for voice
        )
        
        return response.choices[0].message.content.strip()
    
    def get_pricing_info(self, product: str) -> Dict:
        """Retrieve pricing information for products"""
        # This would connect to your actual pricing database
        pricing = {
            "basic": {"price": "$49/mo", "features": ["Core features", "Email support"]},
            "pro": {"price": "$99/mo", "features": ["All basic features", "Priority support", "Advanced analytics"]},
            "enterprise": {"price": "Custom", "features": ["All pro features", "Dedicated support", "Custom integration"]}
        }
        return pricing.get(product.lower(), {})
    
    def schedule_demo(self, user_info: Dict) -> str:
        """Handle demo scheduling"""
        # This would integrate with your calendar system
        return "I'd be happy to schedule a demo! I'll send you a calendar link to choose a time that works for you."
