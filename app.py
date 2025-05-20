import streamlit as st
import requests
from moviepy.editor import VideoFileClip
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import numpy as np
from collections import Counter
import torch
import os
import ffmpeg

@st.cache_resource(show_spinner=False)
def load_models():
    model_name = "ylacombe/accent-classifier"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
    return processor, model

def download_video(url, filename="video.mp4"):
    r = requests.get(url)
    with open(filename, "wb") as f:
        f.write(r.content)

def extract_audio(video_path, audio_path="audio.wav"):
    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le', ac=1, ar='16000')  # mono, 16kHz
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise ValueError(f"Failed to extract audio: {e}")

def split_audio(audio_input, sample_rate, chunk_duration=30):
    """
    تقسيم الصوت إلى مقاطع (كل مقطع مدته chunk_duration ثانية)
    """
    total_samples = len(audio_input)
    chunk_samples = int(chunk_duration * sample_rate)
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        chunks.append(audio_input[start:end])
    return chunks

def predict_accent(audio_path, processor, model):
    audio_input, sample_rate = sf.read(audio_path)
    # تحويل الصوت إلى مونو إذا كان ستيريو
    if len(audio_input.shape) > 1:
        audio_input = audio_input.mean(axis=1)
    # تحويل نوع البيانات إلى float32
    if audio_input.dtype != 'float32':
        audio_input = audio_input.astype('float32')

    # تقسيم الصوت إلى أجزاء كل جزء 30 ثانية
    chunks = split_audio(audio_input, sample_rate, chunk_duration=30)
    results = []
    for chunk in chunks:
        # التأكد من أن العينة 16k
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torch.tensor(chunk, dtype=torch.float32)
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            chunk = resampler(audio_tensor).numpy()
        inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_id = torch.argmax(probabilities).item()
        predicted_label = model.config.id2label[predicted_id]
        confidence = probabilities[0][predicted_id].item()
        results.append((predicted_label, confidence))

    # اختيار الـ accent الأكثر تكرارًا
    labels = [r[0] for r in results]
    most_common_label, _ = Counter(labels).most_common(1)[0]
    # حساب متوسط الثقة لهذا accent فقط
    avg_confidence = np.mean([c for l, c in results if l == most_common_label])

    return most_common_label, avg_confidence
st.title("Accent Analyzer from Video URL")

video_url = st.text_input("Enter video URL (direct link)")

if st.button("Analyze") and video_url:
    try:
        with st.spinner("Downloading video..."):
            download_video(video_url, "video.mp4")

        with st.spinner("Extracting audio..."):
            extract_audio("video.mp4", "audio.wav")

        with st.spinner("Loading models..."):
            processor, model = load_models()

        with st.spinner("Analyzing accent..."):
            accent, confidence = predict_accent("audio.wav", processor, model)

        st.success("Analysis Complete!")
        st.write(f"**Predicted Accent:** {accent}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")

        # تشغيل الصوت داخل التطبيق
        audio_bytes = open("audio.wav", "rb").read()
        st.audio(audio_bytes, format='audio/wav')

    except Exception as e:
        st.error(f"Error: {e}")

