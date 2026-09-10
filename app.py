import os
import time
import uuid
import base64
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ==========================================
# APP IDENTITY
# ==========================================

APP_NAME = "Aria"
APP_TAGLINE = "Your Voice, Understood."

# ==========================================
# LOGO — embedded directly as SVG strings
# (no external files needed; keeps the repo to a single script)
# ==========================================

LOGO_SVG = """
<svg viewBox="0 0 560 200" xmlns="http://www.w3.org/2000/svg">
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
      <feGaussianBlur stdDeviation="7" result="blur"/>
      <feMerge>
        <feMergeNode in="blur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
  </defs>

  <rect width="560" height="200" fill="#0b0715"/>

  <!-- Glowing orb -->
  <g filter="url(#glow)">
    <circle cx="100" cy="100" r="66" fill="url(#orbGrad)"/>
  </g>

  <!-- Subtle inner ring -->
  <circle cx="100" cy="100" r="58" fill="none" stroke="#ffffff" stroke-opacity="0.12" stroke-width="1.5"/>

  <!-- Voice bars (soundwave) -->
  <g stroke="#f5f1ff" stroke-width="6" stroke-linecap="round" opacity="0.95">
    <line x1="68"  y1="92"  x2="68"  y2="108"/>
    <line x1="82"  y1="78"  x2="82"  y2="122"/>
    <line x1="96"  y1="62"  x2="96"  y2="138"/>
    <line x1="110" y1="76"  x2="110" y2="124"/>
    <line x1="124" y1="88"  x2="124" y2="112"/>
  </g>

  <!-- Wordmark "Aria" -->
  <text x="200" y="116"
        font-family="'Inter','Segoe UI',Arial,Helvetica,sans-serif"
        font-size="78" font-weight="900"
        fill="url(#textGrad)"
        letter-spacing="-1">Aria</text>

  <!-- Tagline -->
  <text x="205" y="148"
        font-family="'Inter','Segoe UI',Arial,Helvetica,sans-serif"
        font-size="17" font-weight="500"
        fill="#b8b2d6"
        letter-spacing="2">YOUR VOICE, UNDERSTOOD.</text>

  <!-- Small accent dot -->
  <circle cx="325" cy="110" r="5" fill="#FF6BD6" opacity="0.9"/>
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

  <!-- Rounded background -->
  <rect width="200" height="200" rx="44" fill="#0b0715"/>

  <!-- Glowing orb -->
  <g filter="url(#glow2)">
    <circle cx="100" cy="100" r="74" fill="url(#orbGrad2)"/>
  </g>

  <!-- Subtle inner ring -->
  <circle cx="100" cy="100" r="64" fill="none" stroke="#ffffff" stroke-opacity="0.15" stroke-width="2"/>

  <!-- Voice bars -->
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
    """Convert an inline SVG string to a base64 data URI for embedding in <img> tags."""
    encoded = base64.b64encode(svg_string.strip().encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


LOGO_DATA_URI = svg_to_data_uri(LOGO_SVG)
ICON_DATA_URI = svg_to_data_uri(ICON_SVG)

# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

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
    st.session_state.history = []          # list of {id, time, question, answer}
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
    st.session_state.asr_model = "openai/whisper-large-v3"
if "view" not in st.session_state:
    st.session_state.view = "assistant"    # "assistant" or "history"
if "status" not in st.session_state:
    st.session_state.status = "idle"       # idle | listening | thinking

# ==========================================
# STYLE
# ==========================================

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .stApp {
        background: radial-gradient(circle at top, #1b1030 0%, #0b0715 65%, #050308 100%);
        color: #EDEBFF;
    }

    .aria-header {
        text-align: center;
        padding: 1.2rem 0 0.4rem 0;
    }
    .aria-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #8A6BFF, #FF6BD6, #6BD6FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 1px;
        margin-bottom: 0;
    }
    .aria-tagline {
        color: #b8b2d6;
        font-size: 0.95rem;
        margin-top: -6px;
    }

    .orb-wrap {
        display: flex;
        justify-content: center;
        margin: 1.5rem 0 0.5rem 0;
    }
    .orb {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, #a78bff, #6b3fd9 55%, #2b1256 100%);
        box-shadow: 0 0 30px rgba(150, 100, 255, 0.55), inset 0 0 25px rgba(255,255,255,0.15);
    }
    .orb.listening {
        animation: pulse 1.1s infinite ease-in-out;
        box-shadow: 0 0 55px rgba(255, 100, 220, 0.75), inset 0 0 25px rgba(255,255,255,0.2);
    }
    .orb.thinking {
        animation: spin 2.2s linear infinite, glow 1.4s infinite ease-in-out;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.12); }
    }
    @keyframes spin {
        from { filter: hue-rotate(0deg); }
        to { filter: hue-rotate(360deg); }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 30px rgba(150,100,255,0.5); }
        50% { box-shadow: 0 0 60px rgba(107,214,255,0.8); }
    }

    .status-text {
        text-align: center;
        color: #cfc9f0;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
        letter-spacing: 0.5px;
    }

    div[data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] {
        background: #0d0918;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# GUARD: TOKEN CHECK
# ==========================================

if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.info("Make sure your .env file contains:\n\nHF_TOKEN=your_token")
    st.stop()

client = InferenceClient(provider="auto", api_key=HF_TOKEN)

# ==========================================
# SIDEBAR — NAV + SETTINGS + HISTORY
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
            options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Instruct"],
            index=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Instruct"]
                  .index(st.session_state.model)
                  if st.session_state.model in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "meta-llama/Llama-3.3-70B-Instruct"]
                  else 0,
        )
        st.session_state.asr_model = st.selectbox(
            "Speech recognition model",
            options=["openai/whisper-large-v3", "openai/whisper-large-v3-turbo"],
            index=0 if st.session_state.asr_model == "openai/whisper-large-v3" else 1,
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
    st.caption(f"💬 {len(st.session_state.history)} conversation(s) saved this session")
    if st.button("🗑️ Clear history", use_container_width=True, disabled=len(st.session_state.history) == 0):
        st.session_state.history = []
        st.rerun()

# ==========================================
# HEADER
# ==========================================

if LOGO_DATA_URI:
    st.markdown(
        f"""
        <div class="aria-header">
            <img src="{LOGO_DATA_URI}" style="max-width:340px; width:100%; height:auto;">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f"""
        <div class="aria-header">
            <div class="aria-title">{APP_NAME}</div>
            <div class="aria-tagline">{APP_TAGLINE}</div>
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
# VIEW: ASSISTANT (main)
# ==========================================

orb_class = "orb"
status_label = "Tap the mic and start speaking"
if st.session_state.status == "listening":
    orb_class = "orb listening"
    status_label = "Listening..."
elif st.session_state.status == "thinking":
    orb_class = "orb thinking"
    status_label = "Thinking..."

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
    orb_placeholder.markdown(f'<div class="orb-wrap"><div class="{classes}"></div></div>', unsafe_allow_html=True)
    status_placeholder.markdown(f'<div class="status-text">{label}</div>', unsafe_allow_html=True)

render_orb(st.session_state.status)

audio = st.audio_input("🎤 Speak to Aria", sample_rate=16000)

# Show recent chat bubbles (last 3 exchanges) above the input result
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

    # ---- Speech to text ----
    st.session_state.status = "listening"
    render_orb("listening")

    with st.spinner("🎧 Understanding your voice..."):
        try:
            transcription = client.automatic_speech_recognition(
                audio=audio.getvalue(),
                model=st.session_state.asr_model,
            )
            user_text = transcription.text.strip()
        except Exception as e:
            st.session_state.status = "idle"
            st.error("Speech recognition failed.")
            st.code(str(e))
            st.stop()

    if not user_text:
        st.session_state.status = "idle"
        st.warning("I couldn't understand what you said.")
        st.stop()

    with st.chat_message("user", avatar="🎙️"):
        st.write(user_text)

    # ---- AI response ----
    st.session_state.status = "thinking"
    render_orb("thinking")

    with st.spinner("🤖 Aria is thinking..."):
        try:
            response = client.chat.completions.create(
                model=st.session_state.model,
                messages=[
                    {"role": "system", "content": st.session_state.system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=st.session_state.max_tokens,
                temperature=st.session_state.temperature,
            )
            answer = response.choices[0].message.content
        except Exception as e:
            st.session_state.status = "idle"
            st.error("AI response failed.")
            st.code(str(e))
            st.stop()

    st.session_state.status = "idle"
    render_orb("idle")

    with st.chat_message("assistant", avatar="✨"):
        st.write(answer)

    # ---- Save to history ----
    st.session_state.history.append(
        {
            "id": str(uuid.uuid4()),
            "time": datetime.now().strftime("%b %d, %I:%M %p"),
            "question": user_text,
            "answer": answer,
        }
    )
