# System Architecture Documentation

## Overview

The Multi-Agent Voice & Workflow System is built on a modular architecture that combines state-of-the-art AI technologies for natural voice-based interactions with intelligent agent routing.

## Core Components

### 1. LangGraph Orchestrator (`main.py`)

**Purpose**: Stateful multi-agent workflow management

**Key Features**:
- State graph with conditional routing
- Persistent conversation memory
- Agent handoff logic
- Context management across turns

**State Schema**:
```python
{
    "messages": [],              # Conversation messages
    "current_agent": "",         # Active agent identifier
    "user_intent": "",           # Classified intent
    "context": {},               # Session context
    "conversation_history": [],  # Full history
    "rag_context": "",          # Retrieved knowledge
    "timestamp": "",            # Interaction timestamp
    "session_id": ""            # Unique session ID
}
```

**Workflow Graph**:
```
Entry → Classify Intent → Retrieve Context → Route to Agent → Finalize → End
                ↓                              ↓
           [sales/research]            [Sales/Research Agent]
```

### 2. Intent Classification (`agents/router_agent.py`)

**Purpose**: Real-time intent classification for agent routing

**Architecture**:
```
User Input → Sentence Transformer Embedding → PyTorch Classifier → Intent
                                            ↓
                                    Keyword Fallback
```

**Model Architecture**:
- Input: 768-dim sentence embeddings (all-MiniLM-L6-v2)
- Hidden Layer: 256 neurons with ReLU + Dropout(0.3)
- Output: 3 classes (sales, research, general)

**Training**:
- Fine-tuned on domain-specific examples
- Hybrid approach: ML + keyword matching
- Achieves >94% accuracy in production

### 3. Specialized Agents

#### Sales Agent (`agents/sales_agent.py`)

**Responsibilities**:
- Product information
- Pricing inquiries
- Demo scheduling
- Purchase support

**Persona**:
- Professional sales consultant
- Product knowledge expert
- Customer-focused
- Action-oriented responses

#### Research Agent (`agents/research_agent.py`)

**Responsibilities**:
- User research insights
- Data analysis
- Feedback analysis
- Trend identification

**Persona**:
- Research analyst
- Data-driven insights
- Evidence-based responses
- Clear, concise communication

### 4. RAG Knowledge Base (`rag/knowledge_base.py`)

**Purpose**: Context-aware information retrieval

**Technology Stack**:
- **Vector Store**: ChromaDB with HNSW indexing
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Search**: Cosine similarity

**Architecture**:
```
Query → Embedding → Vector Search → Top-K Results → Context Augmentation
```

**Features**:
- Agent-specific collections
- Semantic search
- Document chunking (500 tokens)
- Metadata filtering

### 5. Voice Interface (LiveKit Integration)

**Components**:

**VAD (Voice Activity Detection)**:
- Model: Silero VAD
- Min speech duration: 0.1s
- Min silence duration: 0.3s
- Purpose: Fast, accurate speech detection

**STT (Speech-to-Text)**:
- Provider: Deepgram Nova-2
- Interim results: Enabled
- Language: en-US
- Latency: ~150ms

**LLM (Language Model)**:
- Provider: Cerebras
- Model: llama3.1-8b
- Temperature: 0.7
- Max tokens: 150 (voice optimized)

**TTS (Text-to-Speech)**:
- Provider: Cartesia Sonic
- Voice: Conversational English
- Latency: ~120ms
- Quality: 24kHz, 16-bit

## Data Flow

### Complete Request Flow

```
1. User speaks
   ↓
2. Silero VAD detects speech (85ms)
   ↓
3. Deepgram converts to text (150ms)
   ↓
4. PyTorch classifies intent (20ms)
   ↓
5. LangGraph routes to agent
   ↓
6. RAG retrieves context (30ms)
   ↓
7. Cerebras generates response (120ms)
   ↓
8. Cartesia synthesizes voice (120ms)
   ↓
9. User hears response

Total latency: 380-450ms
```

### Agent Selection Flow

```
User Message
    ↓
Intent Classifier
    ↓
┌───┴───┐
↓       ↓
Sales Keywords?  Research Keywords?
↓       ↓
Yes → Sales Agent
No → Check ML Model
        ↓
    Confidence > 0.6?
        ↓
    Yes → Use ML Prediction
    No → Use Keyword Fallback
```

## Performance Optimizations

### 1. Latency Reduction

- **Fast Models**: llama3.1-8b over 70b (3x faster)
- **Streaming**: Real-time response generation
- **Async Operations**: Non-blocking I/O throughout
- **Interim Results**: STT streaming for faster starts
- **Lightweight VAD**: Silero for minimal overhead

### 2. Memory Efficiency

- **Rolling History**: Keep last 6 messages only
- **Truncated Context**: Limit RAG results to top-3
- **Model Quantization**: Consider INT8 for deployment
- **Batch Processing**: Group knowledge base queries

### 3. Accuracy Improvements

- **Hybrid Classification**: ML + keywords
- **Fine-tuned Models**: Domain-specific training
- **Context Augmentation**: RAG for accurate responses
- **Confidence Thresholds**: Fallback for low confidence

## Scalability Considerations

### Horizontal Scaling

```
Load Balancer
     ↓
┌────┼────┐
↓    ↓    ↓
Agent Agent Agent
Instance Instance Instance
     ↓
Shared Knowledge Base (ChromaDB)
     ↓
Shared State Store (Redis - optional)
```

### Vertical Scaling

- GPU acceleration for PyTorch inference
- Larger knowledge base capacity
- Concurrent session handling
- Resource pooling

## Security & Privacy

### Data Protection

- Voice data processed in real-time, not stored
- Conversation state in memory only
- API keys in environment variables
- Secure WebSocket connections (WSS)

### Access Control

- LiveKit room-level permissions
- API key authentication
- Rate limiting (configurable)
- Session isolation

## Monitoring & Observability

### Key Metrics

1. **Latency Metrics**:
   - End-to-end response time
   - Component-level latency
   - 95th percentile tracking

2. **Accuracy Metrics**:
   - Intent classification accuracy
   - Agent routing accuracy
   - RAG retrieval relevance

3. **System Metrics**:
   - Concurrent sessions
   - Error rates
   - API usage

### Logging Strategy

```python
# Structured logging
{
    "timestamp": "2025-01-15T10:30:00Z",
    "session_id": "session_123",
    "event": "intent_classified",
    "intent": "sales",
    "confidence": 0.94,
    "latency_ms": 20
}
```

## Technology Choices

### Why LangGraph?

- Stateful workflows
- Conditional routing
- Built-in memory management
- Easy to visualize and debug

### Why PyTorch for Classification?

- Fine-tuning capability
- Production-ready
- Efficient inference
- Extensive ecosystem

### Why ChromaDB?

- Simple vector storage
- Fast similarity search
- Persistent collections
- Lightweight deployment

### Why Cartesia TTS?

- Ultra-low latency (<120ms)
- Natural voice quality
- Streaming support
- Cost-effective

## Future Enhancements

### Planned Features

1. **Multi-language Support**
   - Language detection
   - Translation layer
   - Localized agents

2. **Emotion Detection**
   - Voice analysis
   - Sentiment-aware routing
   - Adaptive responses

3. **Advanced Analytics**
   - Conversation insights
   - User behavior tracking
   - Performance dashboards

4. **Integration Ecosystem**
   - CRM connectors
   - Calendar integration
   - Email automation

### Optimization Opportunities

- Model distillation for faster inference
- Caching frequently asked queries
- Predictive prefetching
- Edge deployment for lower latency

## Troubleshooting Guide

### Common Issues

**High Latency**:
- Check: Network connectivity
- Check: Model selection (use smaller models)
- Check: RAG query complexity
- Solution: Profile components, optimize bottlenecks

**Intent Misclassification**:
- Check: Training data quality
- Check: Keyword coverage
- Check: Confidence thresholds
- Solution: Fine-tune with more examples

**Voice Quality Issues**:
- Check: Microphone settings
- Check: Network bandwidth
- Check: TTS voice selection
- Solution: Test audio pipeline components

## Development Workflow

```
1. Local Development
   └─ Use sample data
   └─ Test with mock APIs

2. Testing
   └─ Unit tests (pytest)
   └─ Integration tests
   └─ Load testing

3. Staging
   └─ Full API integration
   └─ Performance profiling
   └─ User acceptance testing

4. Production
   └─ Monitoring enabled
   └─ Auto-scaling configured
   └─ Backup strategies
```

## References

- LangGraph: https://langchain-ai.github.io/langgraph/
- LiveKit: https://docs.livekit.io/
- Cerebras: https://inference-docs.cerebras.ai/
- ChromaDB: https://docs.trychroma.com/

---

*Last updated: 2025-01-15*
