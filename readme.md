# 🧠 TARA — Your Local AI Intelligence

<div align="center">

**An emotionally aware, fully local AI companion — infinitely extensible and designed to become your personal JARVIS.**

</div>

---

## 📖 Overview

**TARA** (Trusted Autonomous Responsive Assistant) is a Python-based, entirely local AI assistant. It emphasizes **autonomy**, **modularity**, and **privacy** — with no reliance on cloud APIs or paid services.

### Why TARA?

- 🔒 **Privacy-First**: All processing is done locally — your data never leaves your device.
- ⚡ **Low Latency**: Instant, offline responses.
- 🛠️ **Customizable & Extensible**: Change behaviors, add new skills, or extend modules as needed.
- 💸 **Free to Run**: No paywalls or subscriptions, just open-source tools.
- 🧩 **Modular Design**: Architected for easy expansion and plugin of new abilities.

---

## ✨ Core Features

### 🎙️ Voice Interaction
- **Wake Word Detection** — Always listening for your trigger phrase ("Hey TARA" by default)
- **Background Listening** — Smart interrupt handling for mid-conversation stops
- **Local STT/TTS** — Real-time speech recognition and speech synthesis using local models

### 🧠 Local Intelligence
- **Ollama LLM Integration** — Any local LLM supported by Ollama (such as Granite, Mistral, Llama 3, Gemma)
- **Context Awareness** — Maintains conversation context
- **Customizable Personalities** — Adapt tone, style, and behavior

### 🏗️ Architecture
- **Offline** — No external APIs required; runs entirely on your machine
- **Extensible Modules** — Add new skills for tasks, system control, monitoring, code, or home automation
- **Cross-platform** — Works on Windows, macOS, and Linux

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- 8GB+ RAM (recommended)
- GPU (optional, recommended)
- Microphone

### Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Amar2502/Tara.git
   cd Tara
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On Linux / macOS
   source venv/bin/activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama (for LLM)**
   ```bash
   # Linux / macOS
   curl -fsSL https://ollama.com/install.sh | sh
   # Windows: Download from https://ollama.com/download
   ```

5. **Download a Supported LLM Model**
   ```bash
   # Recommended model:
   ollama pull granite4
   # Other options:
   # ollama pull mistral
   # ollama pull gemma2
   ```

6. **Start TARA**
   ```bash
   python main.py
   ```

---

## 🎯 Quick Start

After installation:

```bash
# Start TARA
python main.py

# When prompted, say your wake word (default: "Hey TARA")
# Then speak — TARA will listen and respond!
```

### Example Interactions

```
You: "Hey TARA, what's the weather like?"
TARA: "I can help you check the weather. Would you like me to integrate a weather module?"

You: "Hey TARA, remind me to call John at 3 PM"
TARA: "Alright, I'll remind you to call John at 3 PM."

You: "Hey TARA, explain quantum computing"
TARA: "Quantum computing uses quantum mechanics to process information differently than classical computers..."
```

---

## 🛣️ Roadmap

### 1.0 — Foundation
- [x] Wake word detection
- [x] Speech-to-text and text-to-speech (local)
- [x] Ollama LLM integration
- [x] Basic conversation management

### 2.0 — Intelligence
- [ ] **Context Memory** (history, user prefs)
- [ ] **Emotion Detection** (sentiment, adaptive replies)
- [ ] **Multi-turn Dialogues**
- [ ] **Customizable Wake Words**

### 3.0 — Autonomy
- [ ] **Tool Calling / Module API (MCP)**
- [ ] **Scheduled Tasks**
- [ ] **OS/System Integration** (system control, monitoring)
- [ ] **Background/Daemon Mode**

### 4.0 — JARVIS
- [ ] **IDE Integration** (coding assistance)
- [ ] **Vision/Multimodal**
- [ ] **Voice Cloning**
- [ ] **Adaptive Learning**

---

## 🙏 Acknowledgments

- **Ollama** — Local LLM hosting
- **Faster-Whisper** — Local speech recognition
- **Edge TTS & RealtimeTTS** — Local TTS
- The open-source community

---

## Connect

- **Email**: [amarpandey2502@gmail.com](mailto:amarpandey2502@gmail.com)
- ⭐ Star the repo if you find it useful!

---

<div align="center">

**Built with ❤️ for the future of personal AI**

[⬆ Back to Top](#-tara--your-local-digital-intelligence)

</div>