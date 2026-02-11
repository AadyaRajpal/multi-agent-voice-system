#!/usr/bin/env python3
"""
Multi-Agent Voice & Workflow System
Orchestrates specialized agents using LangGraph with real-time voice interface
"""

import os
import json
import asyncio
from typing import TypedDict, Annotated, Literal, Optional
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    ChatContext,
)
from livekit.plugins import deepgram, openai, silero, cartesia
from cerebras.cloud.sdk import Cerebras

from agents.sales_agent import SalesAgent
from agents.research_agent import ResearchAgent
from agents.router_agent import IntentClassifier
from rag.knowledge_base import KnowledgeBase

# ==================== Configuration ====================
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY", "")


# ==================== State Definition ====================
class AgentState(TypedDict):
    """State shared across all agents in the workflow"""
    messages: list[dict]
    current_agent: str
    user_intent: str
    context: dict
    conversation_history: list[dict]
    rag_context: Optional[str]
    timestamp: str
    session_id: str


# ==================== LangGraph Workflow ====================
class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents using LangGraph"""
    
    def __init__(self, cerebras_api_key: str):
        self.cerebras_client = Cerebras(api_key=cerebras_api_key)
        self.intent_classifier = IntentClassifier()
        self.sales_agent = SalesAgent(cerebras_api_key)
        self.research_agent = ResearchAgent(cerebras_api_key)
        self.knowledge_base = KnowledgeBase()
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=self.memory)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with agent routing"""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each component
        workflow.add_node("classify_intent", self.classify_intent)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("sales_agent", self.sales_agent_node)
        workflow.add_node("research_agent", self.research_agent_node)
        workflow.add_node("finalize_response", self.finalize_response)
        
        # Set entry point
        workflow.set_entry_point("classify_intent")
        
        # Add conditional routing based on intent
        workflow.add_conditional_edges(
            "classify_intent",
            self.route_to_agent,
            {
                "sales": "retrieve_context",
                "research": "retrieve_context",
                "unknown": "finalize_response"
            }
        )
        
        # Route from context retrieval to appropriate agent
        workflow.add_conditional_edges(
            "retrieve_context",
            lambda state: state["current_agent"],
            {
                "sales": "sales_agent",
                "research": "research_agent"
            }
        )
        
        # Both agents route to finalize
        workflow.add_edge("sales_agent", "finalize_response")
        workflow.add_edge("research_agent", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow
    
    async def classify_intent(self, state: AgentState) -> AgentState:
        """Classify user intent using fine-tuned model"""
        last_message = state["messages"][-1]["content"]
        
        # Use intent classifier (PyTorch-based)
        intent = await self.intent_classifier.classify(last_message)
        
        state["user_intent"] = intent
        state["current_agent"] = "sales" if intent == "sales" else "research"
        
        print(f"🎯 Intent classified: {intent} -> Routing to {state['current_agent']} agent")
        return state
    
    async def retrieve_context(self, state: AgentState) -> AgentState:
        """Retrieve relevant context from RAG knowledge base"""
        last_message = state["messages"][-1]["content"]
        agent_type = state["current_agent"]
        
        # Query knowledge base for relevant context
        rag_results = await self.knowledge_base.query(
            query=last_message,
            agent_type=agent_type,
            top_k=3
        )
        
        state["rag_context"] = rag_results
        print(f"📚 Retrieved RAG context: {len(rag_results)} chunks")
        return state
    
    async def sales_agent_node(self, state: AgentState) -> AgentState:
        """Execute sales agent workflow"""
        response = await self.sales_agent.process(
            message=state["messages"][-1]["content"],
            context=state["rag_context"],
            history=state["conversation_history"]
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response,
            "agent": "sales"
        })
        
        print(f"💼 Sales agent response generated")
        return state
    
    async def research_agent_node(self, state: AgentState) -> AgentState:
        """Execute research agent workflow"""
        response = await self.research_agent.process(
            message=state["messages"][-1]["content"],
            context=state["rag_context"],
            history=state["conversation_history"]
        )
        
        state["messages"].append({
            "role": "assistant",
            "content": response,
            "agent": "research"
        })
        
        print(f"🔬 Research agent response generated")
        return state
    
    def route_to_agent(self, state: AgentState) -> str:
        """Route based on classified intent"""
        intent = state["user_intent"]
        if intent in ["sales", "purchase", "pricing", "demo"]:
            return "sales"
        elif intent in ["research", "information", "analysis", "data"]:
            return "research"
        else:
            return "unknown"
    
    async def finalize_response(self, state: AgentState) -> AgentState:
        """Finalize and log the response"""
        state["timestamp"] = datetime.now().isoformat()
        
        # Update conversation history
        state["conversation_history"].extend(state["messages"][-2:])
        
        print(f"✅ Response finalized at {state['timestamp']}")
        return state
    
    async def process_message(self, message: str, session_id: str) -> str:
        """Process a user message through the workflow"""
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": message}],
            "current_agent": "",
            "user_intent": "",
            "context": {},
            "conversation_history": [],
            "rag_context": None,
            "timestamp": "",
            "session_id": session_id
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": session_id}}
        result = await self.app.ainvoke(initial_state, config)
        
        # Extract final response
        final_response = result["messages"][-1]["content"]
        return final_response


# ==================== Voice Agent Integration ====================
class VoiceMultiAgent(Agent):
    """Voice interface for the multi-agent system"""
    
    def __init__(
        self,
        chat_ctx: ChatContext,
        orchestrator: MultiAgentOrchestrator,
        session_id: str
    ):
        self.orchestrator = orchestrator
        self.session_id = session_id
        
        instructions = """
You are an intelligent voice assistant powered by a multi-agent system.
You can help with sales inquiries and research questions.
Keep responses conversational, clear, and under 3 sentences for voice delivery.
You will be routed to specialized agents based on user intent.
"""
        super().__init__(chat_ctx=chat_ctx, instructions=instructions)
    
    async def generate_response(self, user_message: str) -> str:
        """Generate response using the orchestrator"""
        return await self.orchestrator.process_message(user_message, self.session_id)


# ==================== LiveKit Entry Point ====================
async def entrypoint(ctx: JobContext):
    """Main entry point for LiveKit voice interface"""
    
    print("=" * 60)
    print("🚀 Multi-Agent Voice System Starting")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = MultiAgentOrchestrator(CEREBRAS_API_KEY)
    
    # Generate session ID
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize chat context
    chat_ctx = ChatContext()
    
    # Create voice session with optimized settings for <500ms latency
    async with AgentSession(
        vad=silero.VAD.load(
            min_speech_duration=0.1,  # Quick activation
            min_silence_duration=0.3   # Fast cutoff
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="en-US",
            interim_results=True  # Enable streaming for lower latency
        ),
        llm=openai.LLM.with_cerebras(
            model="llama3.1-8b",  # Faster model for lower latency
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            model="sonic-english",  # Ultra-low latency TTS
            voice="248be419-c632-4f23-adf1-5324ed7dbf1d"  # Conversational female voice
        ),
    ) as session:
        
        # Start the session
        agent = VoiceMultiAgent(
            chat_ctx=chat_ctx,
            orchestrator=orchestrator,
            session_id=session_id
        )
        
        await session.start(agent=agent, room=ctx.room)
        
        # Welcome message
        greeting = """
        Hello! I'm your AI assistant with specialized expertise in sales and research.
        How can I help you today?
        """
        
        chat_ctx.add_message(role="assistant", content=greeting)
        await session.speak(greeting.strip())
        
        # Main conversation loop
        try:
            while True:
                # Listen for user input
                user_input = await session.listen()
                
                if user_input:
                    chat_ctx.add_message(role="user", content=user_input)
                    
                    # Process through multi-agent system
                    response = await agent.generate_response(user_input)
                    
                    chat_ctx.add_message(role="assistant", content=response)
                    await session.speak(response)
        
        except Exception as e:
            print(f"❌ Error in conversation loop: {e}")
            error_msg = "I apologize, but I encountered an error. Let's try again."
            await session.speak(error_msg)


# ==================== CLI Entry Point ====================
def main():
    """Run the multi-agent voice system"""
    print("=" * 60)
    print("Multi-Agent Voice & Workflow System")
    print("=" * 60)
    print()
    print("Features:")
    print("  ✓ Stateful multi-agent orchestration with LangGraph")
    print("  ✓ Real-time voice with <500ms latency")
    print("  ✓ Dynamic agent routing based on intent")
    print("  ✓ RAG-powered knowledge retrieval")
    print("  ✓ Specialized sales and research agents")
    print()
    print("Starting system...")
    print("=" * 60)
    
    # Validate configuration
    if not all([LIVEKIT_API_KEY, CEREBRAS_API_KEY, DEEPGRAM_API_KEY, CARTESIA_API_KEY]):
        print("⚠️  Warning: Some API keys are missing!")
        print("Please set all required environment variables.")
        return
    
    # Run the agent
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            ws_url=LIVEKIT_URL,
        )
    )


if __name__ == "__main__":
    main()
