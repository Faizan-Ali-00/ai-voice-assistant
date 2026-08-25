import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# ---------------------------------------
# Load environment variables
# ---------------------------------------

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")


# ---------------------------------------
# Page configuration
# ---------------------------------------

st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)


# ---------------------------------------
# Title
# ---------------------------------------

st.title("🎙️ AI Voice Assistant")

st.write(
    "Speak using your phone microphone and "
    "get an AI response."
)


# ---------------------------------------
# Check Hugging Face token
# ---------------------------------------

if not HF_TOKEN:

    st.error("Hugging Face token not found.")

    st.info(
        "Make sure your .env file contains:\n\n"
        "HF_TOKEN=your_token"
    )

    st.stop()


# ---------------------------------------
# Hugging Face client
# ---------------------------------------

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


# ---------------------------------------
# Microphone
# ---------------------------------------

audio = st.audio_input(
    "🎤 Speak to your assistant",
    sample_rate=16000
)


# ---------------------------------------
# Process voice
# ---------------------------------------

if audio is not None:

    st.audio(audio)

    temp_file = None

    try:

        # Create temporary audio file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as file:

            file.write(audio.getvalue())

            temp_file = file.name


        # ---------------------------------------
        # Speech → Text
        # ---------------------------------------

        with st.spinner("🎧 Understanding your voice..."):

            transcription = client.automatic_speech_recognition(
                audio=temp_file,
                model="openai/whisper-large-v3"
            )

        user_text = transcription.text.strip()


        if not user_text:

            st.warning(
                "I couldn't understand what you said."
            )

            st.stop()


        st.success("Speech recognized!")

        st.write("### 📝 You said:")

        st.write(user_text)


        # ---------------------------------------
        # Text → AI response
        # ---------------------------------------

        with st.spinner("🤖 AI is thinking..."):

            response = client.chat.completions.create(

                model="Qwen/Qwen2.5-7B-Instruct",

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI voice assistant. "
                            "Give clear, concise and useful answers."
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


        # ---------------------------------------
        # Display AI response
        # ---------------------------------------

        st.write("### 🤖 AI:")

        st.write(answer)


    except Exception as error:

        st.error("Something went wrong.")

        st.code(str(error))


    finally:

        # Remove temporary file
        if temp_file and os.path.exists(temp_file):

            os.remove(temp_file)