<div align="center">
  <img src="logos/aria.svg" width="360" alt="Aria Logo" />
</div>

# 🎙️ Aria — Voice Assistant

A beautiful, multi-provider AI voice assistant that listens, thinks, and responds — powered by Whisper speech recognition and a resilient multi-provider fallback chain.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?logo=groq&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-6467F2?logo=openrouter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

Aria is a warm, conversational voice assistant with a modern purple UI. Speak into your mic, and Aria transcribes your voice with Whisper, thinks with a leading LLM, and responds in natural language — all in a smooth, animated interface.

Built with a **multi-provider fallback architecture**, Aria never stops working. If one AI provider runs out of credits or goes down, it automatically switches to the next one in the chain.

## ✨ Features

- 🎙️ Voice input — click the mic and speak naturally
- 🎧 Whisper speech recognition — accurate transcription in 90+ languages
- 🧠 AI chat responses — powered by modern LLMs
- 🔗 Multi-provider fallback — Groq → Gemini → OpenRouter
- 💫 Animated orb — visual feedback for idle / listening / thinking states
- 📌 Sidebar controls — model selection, temperature, tokens, system prompt
- 🕘 Conversation history — session-based with expandable entries
- 🎨 Beautiful purple theme — glowing orb, gradient logo, smooth animations
- 🔒 Secure API keys — stored in Streamlit Secrets, never committed

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Speech Recognition | Whisper Large V3 (via Groq) |
| AI Chat (Primary) | Groq (GPT-OSS 120B / Llama 3.3 70B) |
| AI Chat (Fallback 1) | Google Gemini Flash |
| AI Chat (Fallback 2) | OpenRouter (free models) |
| Language | Python 3.10+ |

## 🔗 Multi-Provider Architecture

Aria uses a provider chain so it never fails due to a single provider running out of credits:

Audio → 1. Groq Whisper → 2. Gemini STT → 3. OpenRouter Whisper

Chat → 1. Groq (Llama / GPT-OSS) → 2. Gemini Flash → 3. OpenRouter (free models)

You only need one provider key to start, but adding all three means zero downtime.

## 📂 Project Structure

ai-voice-assistant/
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── logos/
│   └── aria.svg          # Aria logo (used in README and app)
├── .env                  # API keys (NOT committed)
├── .gitignore            # Git ignore rules
└── README.md

## ⚙️ Installation (Local)

1. Clone the repository

git clone https://github.com/Faizan-Ali-00/ai-voice-assistant.git
cd ai-voice-assistant

2. Create a virtual environment

Windows:
python -m venv venv
venv\Scripts\activate

macOS / Linux:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Set up your API keys

Create a .env file in the root directory:

GROQ_API_KEY=gsk_your_groq_key_here
GEMINI_API_KEY=AIza_your_gemini_key_here
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here

You only need one of these to work. Add all three for maximum resilience.

## 🔑 Getting Free API Keys

| Provider | Free Tier | Get Key |
|----------|-----------|---------|
| Groq | 7,200 audio sec/day · 14,400 req/day | https://console.groq.com/keys |
| Gemini | 15 RPM · 1,500 req/day | https://aistudio.google.com/app/apikey |
| OpenRouter | 20+ free models · 50 req/day | https://openrouter.ai/keys |

## 🚀 Deployment (Streamlit Cloud)

1. Push to GitHub

git add .
git commit -m "Deploy Aria"
git push origin main

2. Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Click New app
3. Select your repo: Faizan-Ali-00/ai-voice-assistant
4. Main file path: app.py
5. Click Deploy

3. Add your API keys as Secrets

Important: Never put API keys in app.py on GitHub — they become public. Use Streamlit Secrets instead.

1. Go to share.streamlit.io → your app → ⋮ → Settings
2. Click the Secrets tab
3. Paste your keys:

GROQ_API_KEY = "gsk_your_groq_key_here"
GEMINI_API_KEY = "AIza_your_gemini_key_here"
OPENROUTER_API_KEY = "sk-or-v1-your_openrouter_key_here"

4. Click Save → Reboot app

## ▶️ Usage

1. Open the app
2. Click the 🎤 mic button
3. Speak your question
4. Click the mic again to stop
5. Aria will transcribe your voice, think, and respond in a chat bubble

### Sidebar Controls

| Setting | Description |
|---------|-------------|
| AI model | Choose GPT-OSS 120B, GPT-OSS 20B, or Qwen |
| Speech recognition model | Whisper Large V3 or Turbo |
| Max tokens | Response length (200 – 4000) |
| Temperature | Creativity level (0.0 – 1.5) |
| System prompt | Customize Aria's personality |

## 🎨 UI Highlights

- Animated orb — pulses when listening, spins when thinking
- Gradient logo — big Aria wordmark with glowing purple orb
- Chat bubbles — clean, rounded, with purple accents
- Sidebar — dark purple theme with toggle arrow
- Provider status — shows which providers are active

## 🔒 Security Notes

- Never commit .env to GitHub
- Always use Streamlit Secrets for deployed apps
- Revoke keys immediately if accidentally exposed
- Store each provider's key separately for easy rotation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: git checkout -b feature/AmazingFeature
3. Commit your changes: git commit -m "Add some AmazingFeature"
4. Push to the branch: git push origin feature/AmazingFeature
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

Faizan Ali
GitHub: https://github.com/Faizan-Ali-00
Repository: https://github.com/Faizan-Ali-00/ai-voice-assistant

## ⭐ Show Your Support

If this project helped you, please give it a star on GitHub.

## 🙏 Acknowledgments

Groq — https://groq.com
Google Gemini — https://ai.google.dev
OpenRouter — https://openrouter.ai
Streamlit — https://streamlit.io
