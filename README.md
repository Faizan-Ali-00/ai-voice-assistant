# 🎙️ AI Voice Assistant

An AI voice assistant built with Streamlit and Hugging Face.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Overview

AI Voice Assistant is a simple, web-based voice assistant powered by Streamlit and Hugging Face models. It provides an interactive interface for voice-based AI conversations.

## ✨ Features

- 🎙️ Voice-based interaction
- 🤖 AI-powered responses via Hugging Face
- 🖥️ Simple Streamlit interface
- ⚡ Fast and lightweight
- 🔌 Modular voice assistant provider routing

## 🛠️ Tech Stack

- Frontend: Streamlit
- AI Models: Hugging Face
- Language: Python 3.10+

## 📂 Project Structure

    ai-voice-assistant/
    ├── app.py                # Streamlit app (main entry point)
    ├── requirements.txt      # Python dependencies
    ├── LICENSE               # MIT License
    ├── .gitignore            # Git ignore rules
    └── README.md

## ⚙️ Installation

### 1. Clone the repository

    git clone https://github.com/Faizan-Ali-00/ai-voice-assistant.git
    cd ai-voice-assistant

### 2. Create a virtual environment

    # Windows
    python -m venv venv
    venv\Scripts\activate

    # macOS / Linux
    python3 -m venv venv
    source venv/bin/activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Set up your API key

Create a `.env` file in the root directory:

    HUGGINGFACE_API_KEY=your_huggingface_api_key_here

Get your API key from https://huggingface.co/settings/tokens

## ▶️ Usage

Run the Streamlit app:

    streamlit run app.py

Then open your browser at http://localhost:8501

1. Speak or type your query
2. Get an AI-generated response
3. Continue the conversation

## 🔒 Notes

- Never commit your `.env` file — it contains your Hugging Face API key
- Make sure `.env` is listed in `.gitignore`
- If you accidentally expose a key, revoke it immediately at https://huggingface.co/settings/tokens

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m "Add some AmazingFeature"`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 👤 Author

Faizan Ali

- GitHub: https://github.com/Faizan-Ali-00
- Repository: https://github.com/Faizan-Ali-00/ai-voice-assistant

## ⭐ Show Your Support

If this project helped you, please give it a star on GitHub.

## 🙏 Acknowledgments

- Hugging Face — https://huggingface.co
- Streamlit — https://streamlit.io
- Open-source contributors
