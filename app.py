import os
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load .env
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
st.write("Speak to your AI assistant using your microphone.")

# Check token
if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.info("Make sure your .env file contains HF_TOKEN=your_token")
    st.stop()

# Hugging Face client
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)

# Microphone
audio = st.audio_input("🎤 Speak to your assistant")

if audio is not None:

    # Speech to text
    with st.spinner("🎧 Understanding your voice..."):

        try:
            result = client.automatic_speech_recognition(
                audio=audio.getvalue(),
                model="openai/whisper-large-v3"
            )

            user_text = result.text

            st.success("Speech recognized!")

            st.write("### 📝 You said:")
            st.write(user_text)

        except Exception as e:
            st.error("Speech recognition failed.")
            st.code(str(e))
            st.stop()

    # AI response
    with st.spinner("🤖 AI is thinking..."):

        try:
            response = client.chat.completions.create(
                model="Qwen/Qwen2.5-7B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI voice assistant. Give clear and concise answers."
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
            st.error("AI response failed.")
            st.code(str(e))