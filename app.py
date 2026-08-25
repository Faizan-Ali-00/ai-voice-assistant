import os
import requests
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

# Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")

# Page settings
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ AI Voice Assistant")
st.write("Speak to your AI assistant using your phone.")

# Check token
if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.stop()


# --------------------------------
# Hugging Face AI client
# --------------------------------

client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


# --------------------------------
# Microphone
# --------------------------------

audio = st.audio_input(
    "🎤 Speak to your assistant",
    sample_rate=16000
)


# --------------------------------
# Process audio
# --------------------------------

if audio is not None:

    st.audio(audio)

    with st.spinner("🎧 Understanding your voice..."):

        try:

            # Get audio bytes
            audio_bytes = audio.getvalue()

            # Hugging Face ASR endpoint
            url = (
                "https://router.huggingface.co/"
                "hf-inference/models/"
                "openai/whisper-large-v3"
            )

            headers = {
                "Authorization": f"Bearer {HF_TOKEN}",
                "Content-Type": "audio/wav"
            }

            # Send audio with explicit MIME type
            response = requests.post(
                url,
                headers=headers,
                data=audio_bytes,
                timeout=120
            )

            # Check response
            if response.status_code != 200:

                st.error("Speech recognition failed.")

                st.code(
                    f"Status: {response.status_code}\n\n"
                    f"{response.text}"
                )

                st.stop()

            result = response.json()

            # Get transcript
            user_text = result.get("text", "")

            if not user_text:
                st.warning("I couldn't understand the audio.")
                st.stop()

            st.success("Speech recognized!")

            st.write("### 📝 You said:")
            st.write(user_text)

        except Exception as e:

            st.error("Speech recognition error.")
            st.code(str(e))

            st.stop()


    # --------------------------------
    # AI response
    # --------------------------------

    with st.spinner("🤖 AI is thinking..."):

        try:

            response = client.chat.completions.create(

                model="Qwen/Qwen2.5-7B-Instruct",

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful AI voice assistant. "
                            "Give clear, simple and concise answers."
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

            st.error("AI response failed.")
            st.code(str(e))