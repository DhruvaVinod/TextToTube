---
title: TextToTube
emoji: 🎥
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.23.3"
app_file: TextToTube.py
pinned: false
---

# TextToTube
## 📖 Overview
TextToTube is an AI-powered educational tool that transforms scanned textbook content into engaging multimedia learning experiences. It bridges the gap between traditional reading and interactive understanding by connecting textbook content with relevant YouTube videos and AI-generated summaries.
## ✨ Features
* Text Scanning : Capture text from textbooks or printed materials using your device camera
* Smart Video Discovery: Find relevant YouTube videos based on scanned content
* AI Summaries: Generate concise summaries of content using Google's Gemini AI
* Multi-language Support: Translate summaries into multiple languages
* Text-to-Speech: Listen to summaries in your preferred language
## 🛠️ Technology Stack
* Computer Vision: OpenCV, EasyOCR, Tesseract OCR
* AI/ML: Google Gemini API, SentenceTransformer
* Speech Processing: Speech Recognition, gTTS
* Video Integration: YouTube Data API
* Frontend: Gradio, Streamlit
* Audio Processing: PyGame, yt-dlp, Whisper
## 🚀 Installation
### Clone the repository:
```bash
git clone https://github.com/DhruvaVinod/TextToTube.git
cd TextToTube
```
 
### Install dependencies: 

```bash
git clone https://github.com/DhruvaVinod/TextToTube.git
cd TextToTube
```
### Set up API keys:
* Create a .env file in the project root
* Add your API keys:
```bash
YOUTUBE_API_KEY=your_youtube_api_key
GEMINI_API_KEY=your_gemini_api_key
```
  
## Screenshots 
## How it works 
* Text Acquisition: Text is captured via camera scan or voice input
* Content Analysis: The extracted text is analyzed using natural language processing
* Video Matching: Semantic search finds the most relevant YouTube videos
* Summary Generation: Gemini AI creates a detailed summary of the video content
* Translation & Speech: The summary is translated and converted to speech
## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
## Acknowledgements

* OpenCV
* EasyOCR
* Tesseract OCR
* Google Gemini
* YouTube Data API
* Gradio
* Whisper
