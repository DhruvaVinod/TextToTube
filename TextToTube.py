import streamlit as st
import cv2
import pytesseract
import numpy as np
import easyocr
import requests
import json
import tempfile
import pygame
from gtts import gTTS
from deep_translator import GoogleTranslator
import webbrowser
import googleapiclient.discovery
from sentence_transformers import SentenceTransformer
import speech_recognition as sr
import io
import os
import time
import gradio as gr

# API Keys
YOUTUBE_API_KEY = "AIzaSyD6hKgUxy-91DW8AnaTrc7nvDHUfWazi_0"
GEMINI_API_KEY = "AIzaSyDDwEucj4KNsnUT4m4qpt1pwnByhm6_vjM"

def scan_headline():
    import cv2
    import easyocr
    import numpy as np
    
    cap = cv2.VideoCapture(0)

    while (True):
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to capture image from camera.")
            break

        cv2.imwrite("captured_frame.jpg", frame)
        print("Image captured. Check 'captured_frame.jpg' to verify.")
        break

    cap.release()
    cv2.destroyAllWindows()
    
    # Convert to grayscale for better OCR accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("processed_frame.jpg", gray)  # Save for debugging

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])
    results = reader.readtext(gray)

    print("OCR Raw Output:", results)  # Debugging line

    if not results:
        print("No text detected. Try adjusting camera focus or lighting.")
        return ""

    extracted_text = " ".join([res[1] for res in results])
    print(extracted_text)
    return extracted_text




from sentence_transformers import SentenceTransformer, util
import googleapiclient.discovery

import webbrowser

def get_video(query):
    import googleapiclient.discovery
    from sentence_transformers import SentenceTransformer
    import numpy as np

    # Load the Sentence Transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Your API Key (Make sure it's enabled for YouTube Data API v3)
    api_key = "AIzaSyD6hKgUxy-91DW8AnaTrc7nvDHUfWazi_0"
    
    # Function to get videos from YouTube API
    def get_top_videos(query, max_results=10):
        try:
            youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=api_key)

            search_response = youtube.search().list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="video"
            ).execute()

            print("\n🔎 Raw API Response:", search_response)  # Debugging Step

            videos = []
            for item in search_response.get("items", []):
                title = item["snippet"]["title"]
                description = item["snippet"]["description"]
                video_id = item["id"]["videoId"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                videos.append({"title": title, "description": description, "url": video_url})

            return videos

        except Exception as e:
            print("\n❌ YouTube API Error:", e)
            return []  # Return empty list if API fails

    # Function to find the best matching video
    def get_best_video(user_query, videos):
        if not videos:
            print("\n No videos found!")
            return None  

        query_embedding = model.encode(user_query, convert_to_tensor=True).cpu().numpy()
        similarities = []

        for video in videos:
            title_embedding = model.encode(video["title"], convert_to_tensor=True).cpu().numpy()
            description_embedding = model.encode(video["description"], convert_to_tensor=True).cpu().numpy()

            title_similarity = np.dot(query_embedding, title_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(title_embedding))
            desc_similarity = np.dot(query_embedding, description_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(description_embedding))

            avg_similarity = (title_similarity + desc_similarity) / 2  # Average similarity
            similarities.append(avg_similarity)

        best_video_index = np.argmax(similarities)
        return videos[best_video_index]

    
    videos = get_top_videos(query)

    if not videos:
        print("\n🚨 No videos returned by API. Possible reasons:\n - Invalid API Key\n - Quota Exceeded\n - Restricted Queries\n - Network Issue")
        return None  # Stop execution if no videos are found

    print("\n✅ Fetched Videos:", videos)  # Debugging

    best_video = get_best_video(query, videos)

    if best_video:
        print("\n🎯 Best Match:", best_video['title'], best_video['url'])
        
        # Open the video in the default web browser
        
        return best_video['title'], best_video['url']
    else:
        return None

def hear_audio(video_url, target_language):
    import os
    import yt_dlp
    from faster_whisper import WhisperModel

    def download_audio(video_url):

        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'no_part': True,
            'force_overwrites': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            file_path = ydl.prepare_filename(info)
            return file_path.rsplit(".", 1)[0] + ".mp3"

    audio_file = download_audio(video_url)

    def transcribe_audio(audio_path):
        model = WhisperModel("base")
        segments, _ = model.transcribe(audio_path)
        text = " ".join([segment.text for segment in segments])
        return text

    transcribed_text = transcribe_audio(audio_file)

    def generate_with_gemini(text: str):
        API_KEY = GEMINI_API_KEY
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"
        prompt = f"Summarize the following news in detail:\n{text}"
        data = {"contents": [{"parts": [{"text": prompt}]}]}
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return "Error generating summary"

    summary = generate_with_gemini(transcribed_text)

    translated_summary = GoogleTranslator(source="auto", target=target_language).translate(summary)

    return translated_summary

def ui_scan_headline():
    text = scan_headline()
    return text

def ui_watch_video(text):
    result = get_video(text)
    if result is None:
        return "No video found", ""
    video_title, video_url = result
    webbrowser.open(video_url)

    return video_title, video_url

def ui_listen_audio(text, target_language):
    result = get_video(text)
    if result is None:
        return "No video found to summarize"
    _, video_url = result
    translated_summary = hear_audio(video_url, target_language)
    return translated_summary


import gradio as gr

# Custom Styling
custom_css = """
body {
    background-image: url('webbg.png'); 
    background-size: cover;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

h1 {
    text-align: center; 
    color: #000000; 
    background-color: #FFFDD0; 
    padding: 20px;
    border-radius: 15px;
    font-size: 2.2rem;
}

.gr-button {
    font-size: 16px;
    font-weight: 600;
    background-color: #add8e6 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px;
}
"""

# UI Layout
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>📰 Textbook to Video & Summary Generator</h1>")

    with gr.Row():
        btn_scan = gr.Button("📷 Scan Text", variant="primary")

    output_text = gr.Textbox(label="📄 Extracted Text", interactive=False)

    with gr.Row():
        btn_watch = gr.Button("▶️ Watch Video", variant="secondary")
        btn_listen = gr.Button("🧾 Generate Summary", variant="secondary")

    video_title = gr.Textbox(label="🎞️ Best Matched Video", interactive=False)
    video_url = gr.Textbox(label="🔗 Video URL", interactive=False)

    with gr.Row():
        language_selector = gr.Dropdown(
            choices=["en", "es", "fr", "de", "hi"],
            label="🌍 Select Language",
            value="en"
        )

    translated_audio_text = gr.Textbox(label="📝 Translated Text Summary", interactive=False)

    btn_scan.click(fn=ui_scan_headline, outputs=output_text)
    btn_watch.click(fn=ui_watch_video, inputs=output_text, outputs=[video_title, video_url])
    btn_listen.click(fn=ui_listen_audio, inputs=[output_text, language_selector], outputs=translated_audio_text)

demo.launch(share=True)
