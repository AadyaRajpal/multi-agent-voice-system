# Multi-Agent Voice & Workflow System 🎙️🤖

A production-ready stateful multi-agent system with real-time voice interface, achieving sub-500ms latency for natural human-AI interactions. Built with LangGraph for orchestration, LiveKit for voice, Cerebras for ultra-fast inference, and PyTorch for intelligent routing.

## 🌟 Key Features

### 🎯 Multi-Agent Orchestration
- **LangGraph Workflow**: Stateful agent orchestration with conditional routing
- **Specialized Agents**: Dedicated sales and research agents with unique personas
- **Dynamic Handoff**: Real-time intent classification for seamless agent switching
- **Conversation Memory**: Persistent state management across interactions

### 🗣️ Voice Interface
- **Sub-500ms Latency**: Optimized for natural conversation flow
- **LiveKit Integration**: Enterprise-grade real-time communication
- **Cartesia TTS**: Ultra-low latency text-to-speech (Sonic model)
- **Deepgram STT**: High-accuracy speech recognition with interim results
- **Silero VAD**: Fast voice activity detection for responsive interactions

### 🧠 Intelligent Routing
- **PyTorch Intent Classifier**: Fine-tuned model for accurate agent routing
- **Hybrid Approach**: Combines ML predictions with keyword fallbacks
- **Real-time Classification**: Instant intent detection for seamless routing
- **Fine-tuning Support**: Easy model training with custom data

### 📚 RAG-Powered Knowledge
- **ChromaDB Vector Store**: Efficient semantic search
- **Agent-Specific Knowledge**: Separate knowledge bases for each agent
- **Context-Aware Responses**: Accurate answers grounded in your data
- **Easy Knowledge Management**: Simple document loading and chunking

### ⚡ Performance Optimized
- **Cerebras Inference**: Ultra-fast LLM responses (llama3.1-8b)
- **Streaming Support**: Real-time response generation
- **Async Architecture**: Non-blocking operations throughout
- **Resource Efficient**: Lightweight models for production deployment

## 🏗️ Architecture

```
User Voice Input
      ↓
[LiveKit + Silero VAD]
      ↓
[Deepgram STT] → Text
      ↓
[LangGraph Orchestrator]
      ↓
[PyTorch Intent Classifier]
      ↓
   ┌──────┴──────┐
   ↓             ↓
[Sales Agent] [Research Agent]
   ↓             ↓
[RAG Knowledge Base Query]
   ↓             ↓
[Cerebras LLM Generation]
   ↓             ↓
   └──────┬──────┘
      ↓
[Response Finalization]
      ↓
[Cartesia TTS] → Voice
      ↓
User Hears Response
```

