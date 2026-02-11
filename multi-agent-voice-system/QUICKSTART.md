# Quick Start Guide 🚀

Get your multi-agent voice system running in under 10 minutes!

## Prerequisites

- Python 3.8 or later
- 4 API keys (all have free tiers)

## Step 1: Get Your API Keys ⚡

### LiveKit (Real-time Communication)
1. Go to https://cloud.livekit.io/
2. Sign up → Create Project
3. Copy: **API Key**, **API Secret**, **WebSocket URL**

### Cerebras (Ultra-Fast LLM)
1. Go to https://inference.cerebras.ai/
2. Sign up → Get API Key
3. Copy: **API Key**

### Deepgram (Speech-to-Text)
1. Go to https://console.deepgram.com/
2. Sign up → Get API Key ($200 free credit)
3. Copy: **API Key**

### Cartesia (Text-to-Speech)
1. Go to https://cartesia.ai/
2. Sign up → Get API Key
3. Copy: **API Key**

## Step 2: Install (One Command) 🛠️

```bash
# Clone the repo
git clone https://github.com/yourusername/multi-agent-voice-system.git
cd multi-agent-voice-system

# Run setup script (Linux/Mac)
chmod +x setup.sh
./setup.sh
```

**Windows users:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Step 3: Configure API Keys 🔑

Edit `.env` file:

```env
LIVEKIT_API_KEY=your_actual_key_here
LIVEKIT_API_SECRET=your_actual_secret_here
LIVEKIT_URL=wss://your-project.livekit.cloud
CEREBRAS_API_KEY=your_actual_key_here
DEEPGRAM_API_KEY=your_actual_key_here
CARTESIA_API_KEY=your_actual_key_here
```

**Important**: Replace all `your_actual_key_here` with real values!

## Step 4: Initialize Knowledge Base 📚

```bash
source venv/bin/activate  # Skip if already activated
python scripts/load_knowledge.py --initialize-sample
```

## Step 5: Run! 🎉

```bash
python main.py
```

You should see:
```
========================================
🚀 Multi-Agent Voice System Starting
========================================
✓ Knowledge base initialized
🎯 System ready - waiting for connections...
```

## Step 6: Connect & Talk 🎙️

1. Open the LiveKit room URL shown in console
2. Allow microphone access
3. Start talking!

**Try these:**
- "How much does the pro plan cost?" → Routes to Sales Agent
- "What do users think about the mobile app?" → Routes to Research Agent
- "Can you schedule a demo for me?" → Routes to Sales Agent

## Troubleshooting 🔧

### "API key missing" error
✅ Check `.env` file has all keys
✅ No quotes around values
✅ Keys are valid (test on provider websites)

### Can't hear the agent
✅ Check speaker volume
✅ Verify Cartesia API key
✅ Try different browser

### Agent can't hear you
✅ Check microphone permissions
✅ Verify Deepgram API key
✅ Test microphone in other apps

### "Failed to connect" error
✅ Check LiveKit URL (must start with `wss://`)
✅ Verify API key and secret match
✅ Check internet connection

## What's Next? 📖

### Customize Your Agents

Edit `agents/sales_agent.py` or `agents/research_agent.py`:

```python
self.persona = """
Your custom instructions here...
"""
```

### Add Your Own Knowledge

```bash
# Create text file
echo "Your knowledge here" > data/sales/my_knowledge.txt

# Load it
python scripts/load_knowledge.py --agent sales --directory data/sales/
```

### Train Intent Classifier

```bash
# Use sample data
python scripts/train_classifier.py --use-sample

# Or your own data
python scripts/train_classifier.py --data my_training_data.json --epochs 20
```

## Architecture Overview 🏗️

```
Voice Input → Deepgram STT → Intent Classifier (PyTorch)
                                      ↓
                          ┌───────────┴───────────┐
                          ↓                       ↓
                    Sales Agent            Research Agent
                          ↓                       ↓
                    Knowledge Base (ChromaDB + RAG)
                          ↓                       ↓
                    Cerebras LLM Generation
                          ↓                       ↓
                    Cartesia TTS → Voice Output
```

**Key Features:**
- ⚡ Sub-500ms latency
- 🎯 94%+ intent classification accuracy
- 📚 RAG-powered knowledge retrieval
- 🔄 Stateful conversation memory
- 🤖 Dynamic agent handoff

## Performance Tips ⚡

For even lower latency:

1. **Use smaller model** (in `main.py`):
```python
model="llama3.1-8b"  # Instead of 70b
```

2. **Adjust VAD settings**:
```python
min_speech_duration=0.05,  # More sensitive
min_silence_duration=0.5    # Longer pause
```

3. **Limit history**:
```python
history[-4:]  # Keep last 2 exchanges instead of 6
```

## Testing 🧪

Run tests to verify everything works:

```bash
# All tests
pytest

# Specific test
pytest tests/test_intent_classifier.py -v

# With coverage
pytest --cov=agents --cov=rag
```

## Production Deployment 🚀

### Using PM2 (recommended)

```bash
npm install -g pm2
pm2 start main.py --name voice-agent --interpreter python3
pm2 save
pm2 startup
```

### Using systemd

Create `/etc/systemd/system/voice-agent.service`:

```ini
[Unit]
Description=Multi-Agent Voice System
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/multi-agent-voice-system
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable voice-agent
sudo systemctl start voice-agent
```

## Common Use Cases 💡

### Customer Support
- Route sales questions to sales agent
- Route technical questions to support agent
- Provide instant responses 24/7

### Sales Enablement
- Answer product questions
- Schedule demos automatically
- Provide pricing information

### User Research
- Analyze feedback in real-time
- Provide data-driven insights
- Answer analytics questions

## Need Help? 🆘

- 📖 Full docs: [README.md](README.md)
- 🏗️ Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/multi-agent-voice-system/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/multi-agent-voice-system/discussions)

## Resources 📚

- [LangGraph Tutorial](https://langchain-ai.github.io/langgraph/tutorials/)
- [LiveKit Agents Guide](https://docs.livekit.io/agents/)
- [Cerebras API Docs](https://inference-docs.cerebras.ai/)
- [PyTorch Fine-tuning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

**🎉 You're all set! Start talking to your AI agents.**

Questions? Open an issue or check the full README.md for details.
