import os
import time
import uuid
import base64
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

# --- Provider SDKs ---
from groq import Groq
from google import genai
from google.genai import types
import httpx
import requests

# ==========================================
# APP IDENTITY
# ==========================================

APP_NAME = "Aria"
APP_TAGLINE = "Your Voice, Understood."

# ==========================================
# LOGO — bigger wordmark, bolder tagline
# ==========================================

LOGO_SVG = """
<svg viewBox="0 0 720 240" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="orbGrad" cx="35%" cy="30%" r="75%">
      <stop offset="0%" stop-color="#d4c5ff"/>
      <stop offset="45%" stop-color="#9a6bff"/>
      <stop offset="80%" stop-color="#6b3fd9"/>
      <stop offset="100%" stop-color="#2b1256"/>
    </radialGradient>
    <linearGradient id="textGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#A78BFF"/>
      <stop offset="50%" stop-color="#FF6BD6"/>
      <stop offset="100%" stop-color="#6BD6FF"/>
    </linearGradient>
    <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="10" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect width="720" height="240" fill="#0b0715" rx="24"/>
  <g filter="url(#glow)">
    <circle cx="130" cy="120" r="82" fill="url(#orbGrad)"/>
  </g>
  <circle cx="130" cy="120" r="72" fill="none" stroke="#ffffff" stroke-opacity="0.14" stroke-width="2"/>
  <g stroke="#f5f1ff" stroke-width="7" stroke-linecap="round" opacity="0.95">
    <line x1="88"  y1="110" x2="88"  y2="130"/>
    <line x1="106" y1="92"  x2="106" y2="148"/>
    <line x1="124" y1="72"  x2="124" y2="168"/>
    <line x1="142" y1="90"  x2="142" y2="150"/>
    <line x1="160" y1="106" x2="160" y2="134"/>
  </g>
  <text x="250" y="135"
        font-family="'Inter','Segoe UI',Arial,Helvetica,sans-serif"
        font-size="110" font-weight="900"
        fill="url(#textGrad)"
        letter-spacing="-3">Aria</text>
  <text x="256" y="180"
        font-family="'Inter','Segoe UI',Arial,Helvetica,sans-serif"
        font-size="18" font-weight="500"
        fill="#b8b2d6"
        letter-spacing="3.5">YOUR VOICE, UNDERSTOOD.</text>
  <circle cx="450" cy="128" r="7" fill="#FF6BD6" opacity="0.95"/>
</svg>
"""

ICON_SVG = """
<svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="orbGrad2" cx="35%" cy="30%" r="75%">
      <stop offset="0%" stop-color="#d4c5ff"/>
      <stop offset="45%" stop-color="#9a6bff"/>
      <stop offset="80%" stop-color="#6b3fd9"/>
      <stop offset="100%" stop-color="#2b1256"/>
    </radialGradient>
    <filter id="glow2" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="9" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>
  <rect width="200" height="200" rx="44" fill="#0b0715"/>
  <g filter="url(#glow2)">
    <circle cx="100" cy="100" r="74" fill="url(#orbGrad2)"/>
  </g>
  <circle cx="100" cy="100" r="64" fill="none" stroke="#ffffff" stroke-opacity="0.15" stroke-width="2"/>
  <g stroke="#f5f1ff" stroke-width="9" stroke-linecap="round" opacity="0.95">
    <line x1="62"  y1="90"  x2="62"  y2="110"/>
    <line x1="80"  y1="70"  x2="80"  y2="130"/>
    <line x1="100" y1="54"  x2="100" y2="146"/>
    <line x1="120" y1="72"  x2="120" y2="128"/>
    <line x1="138" y1="92"  x2="138" y2="108"/>
  </g>
</svg>
"""


def svg_to_data_uri(svg_string: str) -> str:
    encoded = base64.b64encode(svg_string.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


LOGO_DATA_URI = svg_to_data_uri(LOGO_SVG)
ICON_DATA_URI = svg_to_data_uri(ICON_SVG)

# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ==========================================
# PROVIDER FALLBACK SYSTEM
# ==========================================

class MultiProvider:
    """Multi-provider fallback for both STT and chat completions."""

    def __init__(self):
        self.groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
        self.openrouter_key = OPENROUTER_API_KEY

    # ---------- SPEECH-TO-TEXT with fallback ----------
    def transcribe(self, audio_bytes, model="whisper-large-v3"):
        providers = []

        if self.groq_client:
            providers.append(("Groq", self._transcribe_groq))
        if self.gemini_client:
            providers.append(("Gemini", self._transcribe_gemini))
        if self.openrouter_key:
            providers.append(("OpenRouter", self._transcribe_openrouter))

        last_error = None
        for name, func in providers:
            try:
                result = func(audio_bytes, model)
                if result and result.strip():
                    return result, name
            except Exception as e:
                last_error = f"{name}: {e}"
                continue

        raise Exception(f"All STT providers failed. Last error: {last_error}")

    def _transcribe_groq(self, audio_bytes, model):
        transcription = self.groq_client.audio.transcriptions.create(
            file=("audio.wav", audio_bytes),
            model=model,
        )
        return transcription.text

    def _transcribe_gemini(self, audio_bytes, model):
        """Gemini STT via multimodal input [citation:3][citation:17]."""
        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                "Transcribe this audio exactly. Output only the transcript text, nothing else.",
                types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            ],
        )
        return response.text

    def _transcribe_openrouter(self, audio_bytes, model):
        """OpenRouter STT endpoint [citation:13][citation:18]."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        resp = requests.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            headers={
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "openai/whisper-large-v3",
                "input_audio": {"data": audio_b64, "format": "wav"},
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["text"]

    # ---------- CHAT with fallback ----------
    def chat(self, messages, model="openai/gpt-oss-120b", max_tokens=1200, temperature=0.7):
        providers = []

        if self.groq_client:
            providers.append(("Groq", self._chat_groq))
        if self.gemini_client:
            providers.append(("Gemini", self._chat_gemini))
        if self.openrouter_key:
            providers.append(("OpenRouter", self._chat_openrouter))

        last_error = None
        for name, func in providers:
            try:
                result = func(messages, model, max_tokens, temperature)
                if result and result.strip():
                    return result, name
            except Exception as e:
                last_error = f"{name}: {e}"
                continue

        raise Exception(f"All chat providers failed. Last error: {last_error}")

    def _chat_groq(self, messages, model, max_tokens, temperature):
        response = self.groq_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

    def _chat_gemini(self, messages, model, max_tokens, temperature):
        """Gemini chat via OpenAI-compatible endpoint [citation:23]."""
        system_prompt = ""
        user_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                user_messages.append(m)

        contents = []
        if system_prompt:
            contents.append(f"System: {system_prompt}")
        for m in user_messages:
            contents.append(m["content"])

        response = self.gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents="\n\n".join(contents),
        )
        return response.text

    def _chat_openrouter(self, messages, model, max_tokens, temperature):
        """OpenRouter chat completions [citation:13]."""
        # Map Groq model names to OpenRouter free models
        openrouter_model = "openai/gpt-oss-120b:free"
        if "llama" in model.lower():
            openrouter_model = "meta-llama/llama-3.3-70b-instruct:free"

        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": openrouter_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# Initialize multi-provider
provider = MultiProvider()

# Check at least one provider is available
if not any([GROQ_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY]):
    st.error("No AI provider API keys found.")
    st.info("Add at least ONE of these to your .env file:\n\n"
            "GROQ_API_KEY=your_key\n"
            "GEMINI_API_KEY=your_key\n"
            "OPENROUTER_API_KEY=your_key")
    st.stop()

# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title=f"{APP_NAME} — Voice Assistant",
    page_icon=ICON_DATA_URI if ICON_DATA_URI else "🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ==========================================
# SESSION STATE
# ==========================================

if "history" not in st.session_state:
    st.session_state.history = []
if "model" not in st.session_state:
    st.session_state.model = "openai/gpt-oss-120b"
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 1200
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = (
        "You are Aria, a warm, concise voice assistant. "
        "Give complete, useful, accurate answers. Speak naturally, "
        "the way a person would when talking out loud, and don't "
        "stop prematurely."
    )
if "asr_model" not in st.session_state:
    st.session_state.asr_model = "whisper-large-v3"
if "view" not in st.session_state:
    st.session_state.view = "assistant"
if "status" not in st.session_state:
    st.session_state.status = "idle"

# ==========================================
# STYLE
# ==========================================

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent; height: 0;}

    .stApp {
        background: radial-gradient(circle at top, #1b1030 0%, #0b0715 65%, #050308 100%);
        color: #EDEBFF;
    }

    section[data-testid="stSidebar"] {
        background: #0d0918 !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div { color: #d8d3f0; }
    section[data-testid="stSidebar"] small,
    section[data-testid="stSidebar"] .stCaption { color: #8b84b5 !important; }

    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="collapsedControl"] {
        visibility: visible !important;
        opacity: 1 !important;
        color: #d8d3f0 !important;
        z-index: 999 !important;
    }

    .aria-header {
        display: flex;
        justify-content: center;
        padding: 1.5rem 0 0.8rem 0;
    }
    .aria-header img {
        width: 100%;
        max-width: 520px;
        height: auto;
        filter: drop-shadow(0 20px 60px rgba(150, 100, 255, 0.35));
    }

    .orb-wrap {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0 0.5rem 0;
    }
    .orb {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #a78bff, #6b3fd9 55%, #2b1256 100%);
        box-shadow: 0 0 40px rgba(150, 100, 255, 0.6), inset 0 0 30px rgba(255,255,255,0.18);
    }
    .orb.listening {
        animation: pulse 1.1s infinite ease-in-out;
        box-shadow: 0 0 65px rgba(255, 100, 220, 0.8), inset 0 0 30px rgba(255,255,255,0.25);
    }
    .orb.thinking {
        animation: spin 2.2s linear infinite, glow 1.4s infinite ease-in-out;
    }
    @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.12); } }
    @keyframes spin { from { filter: hue-rotate(0deg); } to { filter: hue-rotate(360deg); } }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 40px rgba(150,100,255,0.5); }
        50% { box-shadow: 0 0 70px rgba(107,214,255,0.85); }
    }

    .status-text {
        text-align: center;
        color: #cfc9f0;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        letter-spacing: 0.6px;
    }

    .recorder-title {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 700;
        color: #e8e3ff;
        margin: 1rem 0 0.3rem 0;
        letter-spacing: 0.3px;
    }
    .recorder-hint {
        text-align: center;
        font-size: 0.82rem;
        color: #8b84b5;
        margin-bottom: 1.2rem;
    }

    div[data-testid="stAudioInput"] {
        background: linear-gradient(180deg, rgba(155,107,255,0.09) 0%, rgba(107,63,217,0.05) 100%);
        border: 1px solid rgba(155, 107, 255, 0.28);
        border-radius: 20px;
        padding: 1.4rem 1.6rem !important;
        margin: 0 auto !important;
        max-width: 520px !important;
        box-shadow: 0 12px 40px rgba(107, 63, 217, 0.22);
    }
    div[data-testid="stAudioInput"]:hover { border-color: rgba(155, 107, 255, 0.55); }
    div[data-testid="stAudioInput"] button {
        width: 52px !important;
        height: 52px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #8A6BFF 0%, #6b3fd9 100%) !important;
        border: none !important;
        box-shadow: 0 6px 22px rgba(138, 107, 255, 0.55) !important;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.045);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
        padding: 0.9rem 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:
    if ICON_DATA_URI:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:-6px;">
                <img src="{ICON_DATA_URI}" style="width:38px; height:38px; border-radius:10px;">
                <span style="font-size:1.3rem; font-weight:800;
                             background:linear-gradient(90deg,#8A6BFF,#FF6BD6,#6BD6FF);
                             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
                    {APP_NAME}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"## 🎙️ {APP_NAME}")
    st.caption(APP_TAGLINE)
    st.divider()

    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("🏠 Assistant", use_container_width=True,
                      type="primary" if st.session_state.view == "assistant" else "secondary"):
            st.session_state.view = "assistant"
            st.rerun()
    with nav_col2:
        if st.button("🕘 History", use_container_width=True,
                      type="primary" if st.session_state.view == "history" else "secondary"):
            st.session_state.view = "history"
            st.rerun()

    st.divider()

    with st.expander("⚙️ Settings", expanded=(st.session_state.view == "assistant")):
        st.session_state.model = st.selectbox(
            "AI model",
            options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"],
            index=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
                  .index(st.session_state.model)
                  if st.session_state.model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]
                  else 0,
        )
        st.session_state.asr_model = st.selectbox(
            "Speech recognition model",
            options=["whisper-large-v3", "whisper-large-v3-turbo"],
            index=["whisper-large-v3", "whisper-large-v3-turbo"]
                  .index(st.session_state.asr_model)
                  if st.session_state.asr_model in ["whisper-large-v3", "whisper-large-v3-turbo"]
                  else 0,
        )
        st.session_state.max_tokens = st.slider(
            "Max response length (tokens)", min_value=200, max_value=4000,
            value=st.session_state.max_tokens, step=100,
        )
        st.session_state.temperature = st.slider(
            "Creativity (temperature)", min_value=0.0, max_value=1.5,
            value=st.session_state.temperature, step=0.1,
        )
        st.session_state.system_prompt = st.text_area(
            "Personality / system prompt",
            value=st.session_state.system_prompt,
            height=100,
        )
        st.caption(f"Model: `{st.session_state.model}`")

    st.divider()

    # Show which providers are active
    active = []
    if GROQ_API_KEY: active.append("Groq")
    if GEMINI_API_KEY: active.append("Gemini")
    if OPENROUTER_API_KEY: active.append("OpenRouter")
    st.caption(f"🔗 Providers: {', '.join(active) if active else 'None'}")

    st.caption(f"💬 {len(st.session_state.history)} conversation(s) saved this session")
    if st.button("🗑️ Clear history", use_container_width=True, disabled=len(st.session_state.history) == 0):
        st.session_state.history = []
        st.rerun()

# ==========================================
# HEADER — big logo
# ==========================================

st.markdown(
    f"""
    <div class="aria-header">
        <img src="{LOGO_DATA_URI}" alt="{APP_NAME} logo" />
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# VIEW: HISTORY
# ==========================================

if st.session_state.view == "history":
    st.subheader("🕘 Conversation history")

    if not st.session_state.history:
        st.info("No conversations yet. Go to **Assistant** and speak to Aria to get started.")
    else:
        for entry in reversed(st.session_state.history):
            with st.expander(f"🕐 {entry['time']}  —  \"{entry['question'][:60]}\""):
                with st.chat_message("user", avatar="🎙️"):
                    st.write(entry["question"])
                with st.chat_message("assistant", avatar="✨"):
                    st.write(entry["answer"])

    st.stop()

# ==========================================
# VIEW: ASSISTANT
# ==========================================

orb_placeholder = st.empty()
status_placeholder = st.empty()


def render_orb(status: str):
    classes = "orb"
    label = "Tap the mic and start speaking"
    if status == "listening":
        classes = "orb listening"
        label = "Listening..."
    elif status == "thinking":
        classes = "orb thinking"
        label = "Thinking..."
    orb_placeholder.markdown(
        f'<div class="orb-wrap"><div class="{classes}"></div></div>',
        unsafe_allow_html=True,
    )
    status_placeholder.markdown(
        f'<div class="status-text">{label}</div>',
        unsafe_allow_html=True,
    )


render_orb(st.session_state.status)

st.markdown(
    '<div class="recorder-title">🎤 Speak to Aria</div>'
    '<div class="recorder-hint">Click the mic · Speak · Click again to send</div>',
    unsafe_allow_html=True,
)

audio = st.audio_input(
    "Speak to Aria",
    sample_rate=16000,
    label_visibility="collapsed",
)

if st.session_state.history:
    st.markdown("#### Recent")
    for entry in st.session_state.history[-3:]:
        with st.chat_message("user", avatar="🎙️"):
            st.write(entry["question"])
        with st.chat_message("assistant", avatar="✨"):
            st.write(entry["answer"])

# ==========================================
# PROCESS AUDIO
# ==========================================

if audio is not None:
    st.audio(audio)

    st.session_state.status = "listening"
    render_orb("listening")

    with st.spinner("🎧 Understanding your voice..."):
        try:
            user_text, stt_provider = provider.transcribe(
                audio.getvalue(),
                st.session_state.asr_model,
            )
        except Exception as e:
            st.session_state.status = "idle"
            st.error("Speech recognition failed on all providers.")
            st.code(str(e))
            st.stop()

    if not user_text:
        st.session_state.status = "idle"
        st.warning("I couldn't understand what you said.")
        st.stop()

    with st.chat_message("user", avatar="🎙️"):
        st.write(user_text)

    st.session_state.status = "thinking"
    render_orb("thinking")

    with st.spinner("🤖 Aria is thinking..."):
        try:
            answer, chat_provider = provider.chat(
                messages=[
                    {"role": "system", "content": st.session_state.system_prompt},
                    {"role": "user", "content": user_text},
                ],
                model=st.session_state.model,
                max_tokens=st.session_state.max_tokens,
                temperature=st.session_state.temperature,
            )
        except Exception as e:
            st.session_state.status = "idle"
            st.error("AI response failed on all providers.")
            st.code(str(e))
            st.stop()

    st.session_state.status = "idle"
    render_orb("idle")

    with st.chat_message("assistant", avatar="✨"):
        st.write(answer)

    st.session_state.history.append(
        {
            "id": str(uuid.uuid4()),
            "time": datetime.now().strftime("%b %d, %I:%M %p"),
            "question": user_text,
            "answer": answer,
        }
    )
