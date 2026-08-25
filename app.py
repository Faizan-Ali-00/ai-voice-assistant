import os

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ==========================================
# LOAD ENVIRONMENT
# ==========================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🎙️ AI Voice Assistant")

st.write(
    "Speak to the assistant using your microphone."
)


# ==========================================
# CHECK HUGGING FACE TOKEN
# ==========================================

if not HF_TOKEN:

    st.error("Hugging Face token not found.")

    st.info(
        "Make sure your .env file contains:\n\n"
        "HF_TOKEN=your_token"
    )

    st.stop()


# ==========================================
# HUGGING FACE CLIENT
# ==========================================

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


# ==========================================
# MICROPHONE
# ==========================================

audio = st.audio_input(
    "🎤 Speak to your assistant",
    sample_rate=16000
)


# ==========================================
# PROCESS AUDIO
# ==========================================

if audio is not None:

    # Show recorded audio
    st.audio(audio)

    # ======================================
    # SPEECH TO TEXT
    # ======================================

    with st.spinner("🎧 Understanding your voice..."):

        try:

            transcription = client.automatic_speech_recognition(
                audio=audio.getvalue(),
                model="openai/whisper-large-v3"
            )

            user_text = transcription.text.strip()

        except Exception as e:

            st.error("Speech recognition failed.")

            st.code(str(e))

            st.stop()


    # ======================================
    # CHECK TRANSCRIPTION
    # ======================================

    if not user_text:

        st.warning(
            "I couldn't understand what you said."
        )

        st.stop()


    st.success("Speech recognized!")

    st.write("### 📝 You said:")

    st.write(user_text)


    # ======================================
    # AI RESPONSE
    # ======================================

    with st.spinner("🤖 AI is thinking..."):

        try:

            response = client.chat.completions.create(

                model="openai/gpt-oss-120b",

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI voice assistant. "
                            "Give complete, useful and accurate answers. "
                            "Explain things clearly and do not stop "
                            "prematurely."
                        )
                    },

                    {
                        "role": "user",
                        "content": user_text
                    }
                ],

                max_tokens=7000
            )


            answer = response.choices[0].message.content


        except Exception as e:

            st.error("AI response failed.")

            st.code(str(e))

            st.stop()


    # ======================================
    # SHOW AI RESPONSE
    # ======================================

    st.write("### 🤖 AI:")

    st.write(answer)