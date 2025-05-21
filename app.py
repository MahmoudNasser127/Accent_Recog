import os
import subprocess
import time
from collections import Counter
from typing import Tuple, List
import numpy as np
import requests
import soundfile as sf
import streamlit as st
import torch
import yt_dlp
import imageio_ffmpeg
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
import random

# Constants
MODEL_NAME = "ylacombe/accent-classifier"
VIDEO_FILENAME = "video.mp4"
AUDIO_FILENAME = "audio.wav"
CHUNK_DURATION_SECONDS = 10
SELECTED_IDS = [0, 1, 7, 13, 17, 25]  # American, Australian, English, Indian, Latin American, South African


@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[Wav2Vec2FeatureExtractor, AutoModelForAudioClassification]:
    """Load the feature extractor and model from Hugging Face."""
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
    return feature_extractor, model


def download_video(url: str, filename: str = VIDEO_FILENAME) -> str:
    """Download video from YouTube or direct URL."""
    if "youtube.com" in url or "youtu.be" in url:
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
            'outtmpl': filename,
            'quiet': True,
            'merge_output_format': 'mp4'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    else:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return filename


def extract_audio(video_path: str, audio_path: str = AUDIO_FILENAME) -> str:
    """Extract mono, 16kHz audio from video using ffmpeg."""
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_bin,
        "-i", video_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        audio_path,
        "-y",
    ]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()}")
    return audio_path


def split_audio(audio_input: np.ndarray, sample_rate: int, chunk_duration: int = CHUNK_DURATION_SECONDS) -> List[np.ndarray]:
    """Split audio into chunks of chunk_duration seconds."""
    total_samples = len(audio_input)
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = [audio_input[start:start + chunk_samples] for start in range(0, total_samples, chunk_samples)]
    return chunks


def predict_accent(audio_path: str, feature_extractor, model) -> Tuple[str, float]:
    """Predict accent from audio file."""
    start_time = time.time()

    id2label = model.config.id2label
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    audio_input, sample_rate = sf.read(audio_path)
    if audio_input.ndim > 1:
        audio_input = audio_input.mean(axis=1)
    audio_input = audio_input.astype('float32')

    chunks = split_audio(audio_input, sample_rate)

    if len(chunks) >= 3:
        chunks = random.sample(chunks, 3)
    elif not chunks:
        raise ValueError("Audio too short to extract any valid Accent.")

    results = []
    for chunk in chunks:
        inputs = feature_extractor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits

        logits_filtered = logits[:, SELECTED_IDS]
        probabilities = torch.nn.functional.softmax(logits_filtered, dim=-1)

        predicted_index = torch.argmax(probabilities).item()
        predicted_id = SELECTED_IDS[predicted_index]
        predicted_label = id2label[predicted_id]
        confidence = probabilities[0][predicted_index].item()

        results.append((predicted_label, confidence))

    labels = [r[0] for r in results]
    most_common_label, _ = Counter(labels).most_common(1)[0]
    avg_confidence = np.mean([c for l, c in results if l == most_common_label])

    end_time = time.time()
    st.write(f"⏱️ Prediction took: {end_time - start_time:.2f} seconds")

    return most_common_label, avg_confidence


# Streamlit UI
st.title("Accent Analyzer")

video_url = st.text_input("Enter video URL (direct mp4 link):")

if st.button("Analyze") and video_url:
    try:
        st.info("Downloading video...")
        video_file = download_video(video_url, VIDEO_FILENAME)

        st.info("Extracting audio...")
        audio_file = extract_audio(video_file, AUDIO_FILENAME)

        st.info("Loading model...")
        processor, model = load_models()

        st.info("Analyzing accent...")
        accent, confidence = predict_accent(audio_file, processor, model)

        st.success("Analysis Complete!")
        st.write(f"**Predicted Accent:** {accent}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")

        with open(audio_file, "rb") as f:
            st.audio(f.read(), format='audio/wav')
        summary = (
            f"We looked at different parts of the audio to figure out the speaker’s accent. "
            f"The model thinks it’s **{accent}**, and it’s about **{confidence*100:.2f}%** sure about that. "
            "Keep in mind, this isn’t perfect, but it gives a good idea of the speaker’s English accent."
            )
st.write(summary)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        for file in [VIDEO_FILENAME, AUDIO_FILENAME]:
            if os.path.exists(file):
                os.remove(file)
