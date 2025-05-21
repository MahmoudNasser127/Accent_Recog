# 🎙️ Accent Analyzer Tool

This is a simple Streamlit-based AI tool that accepts a public video URL, extracts the audio, and predicts the English accent of the speaker, along with a confidence score. It's designed as a practical demo for spoken language evaluation and screening.

---

## ✅ Features

- 🔗 **Supports**:  
  - Direct `.mp4` video links (fast download)  
  - YouTube video links (via `yt-dlp`)  
- 🔊 Extracts audio using **FFmpeg**
- 🧠 Classifies accents using a **Hugging Face pretrained model**  
  (Reduced set of accents for faster prediction: e.g., American, British, Indian)
- 🎯 Outputs predicted **accent** and **confidence score**
- 🎧 Allows you to play extracted audio directly in the app

---

## ⚡ Optimization Note

To reduce inference time and improve responsiveness, the number of accent classes used in the model was reduced. This accelerates prediction while maintaining meaningful results for common English accents.

---

## 📦 Requirements

- Python **3.8+**
- FFmpeg installed and accessible in your **system PATH**
- Python packages:
  ```bash
  pip install streamlit requests soundfile torch transformers imageio-ffmpeg yt-dlp numpy
  ```

---

## ⚙️ Installation

1. **Clone the repo** or download the script
2. **Install dependencies** as shown above
3. **Install FFmpeg**:
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add it to your system `PATH`

---

## 🚀 Usage

Start the app:

```bash
streamlit run your_script_name.py
```

In the web app:

1. Enter a public video URL (YouTube or direct `.mp4`)
2. Click **Analyze**
3. The app will:
   - Download the video
   - Extract and play audio
   - Predict accent
   - Display confidence score

---

## 📽️ Video Link Support & Limitations

This tool supports video downloads from:

✅ **Direct .mp4 links**  
✅ **YouTube links** via `yt-dlp`

🚫 **Loom support is limited**:  
Standard Loom share links are encrypted and **do not allow direct download** or audio extraction with FFmpeg. A Loom business account is required to access direct `.mp4` links.

🛠️ Therefore, **testing was done using YouTube links** to demonstrate full functionality of the AI agent.

---

## 🛠️ Troubleshooting

- ❌ **Video download failed?**  
  - Check if it's a direct link or supported YouTube video
  - Ensure your internet connection is active

- ⚠️ **FFmpeg error?**  
  - Make sure FFmpeg is correctly installed and in your system's `PATH`

- 🐍 **Python package error?**  
  - Verify all dependencies are installed via pip

---

## 📬 Contact

Feel free to reach out for questions or suggestions. Pull requests are welcome!
👤 Contact
Mahmoud Nasser
📧 Email: 44mahmoudnasser@gmail.com
🔗 LinkedIn: linkedin.com/in/44mahmoudnasser
---
