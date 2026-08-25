import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Get Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Page settings
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)

# Title
st.title("🎙️ AI Voice Assistant")
st.write("Speak to your AI assistant using your phone microphone.")

# Check token
if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.stop()

# Hugging Face client
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

# Microphone
audio = st.audio_input("🎤 Speak to your assistant")

if audio is not None:

    # Get audio format from browser
    audio_type = audio.type or "audio/wav"

    if "webm" in audio_type:
        extension = ".webm"
    elif "mpeg" in audio_type or "mp3" in audio_type:
        extension = ".mp3"
    elif "ogg" in audio_type:
        extension = ".ogg"
    else:
        extension = ".wav"

    # Save recording as a real audio file
    temp_file = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as f:
            f.write(audio.getvalue())
            temp_file = f.name

        # Show recorded audio
        st.audio(audio)

        # Speech to text
        with st.spinner("🎧 Understanding your voice..."):

            result = client.automatic_speech_recognition(
                temp_file,
                model="openai/whisper-large-v3"
            )

            user_text = result.text

        st.success("Speech recognized!")

        st.write("### 📝 You said:")
        st.write(user_text)

        # AI response
        with st.spinner("🤖 AI is thinking..."):

            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI voice assistant. "
                            "Give clear and concise answers."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],
                max_tokens=300
            )

            answer = response.choices[0].message.content

        st.write("### 🤖 AI:")
        st.write(answer)

    except Exception as e:
        st.error("Something went wrong.")
        st.code(str(e))

    finally:
        # Delete temporary audio file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)