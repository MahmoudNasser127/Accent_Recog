import streamlit as st
import requests
import soundfile as sf
from transformers import AutoProcessor, AutoModelForAudioClassification
from transformers import pipeline, Wav2Vec2FeatureExtractor
import numpy as np
from collections import Counter
import torch
import imageio_ffmpeg
import subprocess
import os
import time

# Load accent classification model
@st.cache_resource(show_spinner=False)
def load_accent_model():
    model_name = "ylacombe/accent-classifier"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    return feature_extractor, model

# Load ASR and summarization pipelines
@st.cache_resource(show_spinner=False)
def load_asr_and_summarization_models():
    # ASR - Whisper large
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-large")
    # Summarization - fine-tuned on podcasts
    summarizer = pipeline("summarization", model="paulowoicho/t5-podcast-summarisation")
    return asr, summarizer

def download_video(url, filename="video.mp4"):
    if "youtube.com" in url or "youtu.be" in url:
        import yt_dlp
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
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return filename

def extract_audio(video_path, audio_path="audio.wav"):
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

def split_audio(audio_input, sample_rate, chunk_duration=30):
    total_samples = len(audio_input)
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunks.append(audio_input[start:end])
    return chunks

def predict_accent(audio_path, feature_extractor, model):
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    audio_input, sample_rate = sf.read(audio_path)
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)
    if audio_input.dtype != 'float32':
        audio_input = audio_input.astype('float32')
    chunks = split_audio(audio_input, sample_rate, chunk_duration=10)
    chunks = chunks[:3]
    results = []
    for chunk in chunks:
        inputs = feature_extractor(chunk, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities).item()
        predicted_label = model.config.id2label[predicted_id]
        confidence = probabilities[0][predicted_id].item()
        results.append((predicted_label, confidence))
    labels = [r[0] for r in results]
    most_common_label, _ = Counter(labels).most_common(1)[0]
    avg_confidence = np.mean([c for l, c in results if l == most_common_label])
    end_time = time.time()
    st.write(f"⏱️ Accent prediction took: {end_time - start_time:.2f} seconds")
    return most_common_label, avg_confidence

def transcribe_and_summarize(audio_path, asr, summarizer):
    st.info("Transcribing audio...")
    transcription_result = asr(audio_path)
    transcript = transcription_result.get("text", "")
    st.text_area("Transcription", transcript, height=150)
    st.info("Summarizing transcription...")
    # For very long transcripts, you might want to chunk it before summarizing
    summary = summarizer(transcript, min_length=30, max_length=130)[0]['summary_text']
    return summary

# Streamlit UI
st.title("Accent Analyzer + Audio Summarizer")

video_url = st.text_input("Enter video URL (direct mp4 link or YouTube):")

if st.button("Analyze") and video_url:
    try:
        st.info("Downloading video...")
        video_file = download_video(video_url, "video.mp4")

        st.info("Extracting audio...")
        audio_file = extract_audio(video_file, "audio.wav")

        st.info("Loading models...")
        accent_feature_extractor, accent_model = load_accent_model()
        asr_pipeline, summarizer_pipeline = load_asr_and_summarization_models()

        st.info("Analyzing accent...")
        accent, confidence = predict_accent(audio_file, accent_feature_extractor, accent_model)
        st.success("Accent Analysis Complete!")
        st.write(f"**Predicted Accent:** {accent}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        summary = transcribe_and_summarize(audio_file, asr_pipeline, summarizer_pipeline)
        st.success("Summary Complete!")
        st.write(f"### Summary:\n{summary}")

        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format='audio/wav')

        os.remove(video_file)
        os.remove(audio_file)

    except Exception as e:
        st.error(f"Error: {e}")
