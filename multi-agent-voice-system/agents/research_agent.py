"""
Research Agent - Specialized for user research, data analysis, and insights
"""

import json
from typing import List, Dict, Optional
from cerebras.cloud.sdk import Cerebras


class ResearchAgent:
    """Handles research inquiries and provides data-driven insights"""
    
    def __init__(self, api_key: str):
        self.client = Cerebras(api_key=api_key)
        self.persona = """
You are a research analyst specializing in user research and data insights.
You provide clear, evidence-based answers with actionable insights.
You're thorough but concise, perfect for voice-based communication.
Keep responses focused and under 3 sentences for voice delivery.
"""
    
    async def process(
        self,
        message: str,
        context: Optional[str] = None,
        history: List[Dict] = None
    ) -> str:
        """Process a research-related message"""
        
        # Build conversation context
        messages = [{"role": "system", "content": self.persona}]
        
        # Add conversation history
        if history:
            messages.extend(history[-6:])  # Last 3 exchanges
        
        # Add RAG context if available
        if context:
            context_prompt = f"""
Relevant research data and insights:
{context}

Use this information to provide accurate, data-driven responses.
"""
            messages.append({"role": "system", "content": context_prompt})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Generate response
        response = self.client.chat.completions.create(
            model="llama3.1-8b",
            messages=messages,
            temperature=0.6,  # Slightly lower for more factual responses
            max_tokens=150
        )
        
        return response.choices[0].message.content.strip()
    
    async def analyze_user_feedback(self, feedback: List[str]) -> Dict:
        """Analyze user feedback and extract insights"""
        analysis_prompt = f"""
Analyze the following user feedback and provide:
1. Key themes
2. Sentiment distribution
3. Top 3 actionable insights

Feedback:
{json.dumps(feedback, indent=2)}
"""
        
        response = self.client.chat.completions.create(
            model="llama3.1-70b",  # Use larger model for analysis
            messages=[
                {"role": "system", "content": "You are a data analyst expert."},
                {"role": "user", "content": analysis_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        return json.loads(response.choices[0].message.content)
    
    async def get_user_insights(self, segment: str) -> Dict:
        """Retrieve insights about a user segment"""
        # This would connect to your analytics database
        insights = {
            "demographics": {},
            "behavior_patterns": [],
            "pain_points": [],
            "opportunities": []
        }
        return insights
