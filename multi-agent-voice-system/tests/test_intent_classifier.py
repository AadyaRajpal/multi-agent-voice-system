"""
Unit tests for intent classification
"""

import pytest
import asyncio
from agents.router_agent import IntentClassifier


@pytest.fixture
def classifier():
    """Initialize intent classifier"""
    return IntentClassifier()


@pytest.mark.asyncio
async def test_sales_intent_classification(classifier):
    """Test sales intent detection"""
    sales_queries = [
        "How much does the pro plan cost?",
        "I want to schedule a demo",
        "What's included in the pricing?",
        "Can I get a discount?",
    ]
    
    for query in sales_queries:
        intent = await classifier.classify(query)
        assert intent == "sales", f"Failed to classify '{query}' as sales"


@pytest.mark.asyncio
async def test_research_intent_classification(classifier):
    """Test research intent detection"""
    research_queries = [
        "What do users think about the mobile app?",
        "Show me the latest survey results",
        "What are the main user pain points?",
        "Can you analyze user feedback?",
    ]
    
    for query in research_queries:
        intent = await classifier.classify(query)
        assert intent == "research", f"Failed to classify '{query}' as research"


@pytest.mark.asyncio
async def test_general_intent_classification(classifier):
    """Test general intent detection"""
    general_queries = [
        "Hello, how are you?",
        "What can you help me with?",
        "Tell me about yourself",
    ]
    
    for query in general_queries:
        intent = await classifier.classify(query)
        assert intent == "general", f"Failed to classify '{query}' as general"


def test_keyword_classification(classifier):
    """Test keyword-based fallback"""
    # Sales keywords
    sales_text = "I need pricing information and want to purchase"
    result = classifier._keyword_classify(sales_text)
    assert result == "sales"
    
    # Research keywords
    research_text = "What does the user research data say about trends"
    result = classifier._keyword_classify(research_text)
    assert result == "research"
    
    # No clear keywords
    general_text = "This is just a random message"
    result = classifier._keyword_classify(general_text)
    assert result == "general"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
