# 🎵 Music Genre Detection API

An end-to-end machine learning system that classifies music genres from audio files using classical ML techniques and serves predictions through a modern web interface.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?logo=fastapi)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)

- 📓 **Training Notebook**: [View on Google Colab](https://colab.research.google.com/drive/1mjXtjz6gdUjcSzuWeIBeLt0GaKeyU-N5?usp=sharing)
- 🌐 **Try Out**: [Vist Here](https://music-genre-detection.vercel.app/)

## 🌟 Overview

This project demonstrates a complete machine learning pipeline from data preprocessing to deployment. The system extracts audio features using librosa, trains multiple classical ML models, and serves the best-performing model (SVM) through a FastAPI backend with a React frontend.

**Live Demo:**
- 🚀 **User Interface**: [https://music-genre-detection.vercel.app/](https://music-genre-detection.vercel.app/)
- 🚀 **API**: [https://music-genre-detection-api-yv06.onrender.com](https://music-genre-detection-api-yv06.onrender.com)
- 📖 **API Documentation**: [https://music-genre-detection-api-yv06.onrender.com/docs](https://music-genre-detection-api-yv06.onrender.com/docs)

## ✨ Features

- 🎼 **10 Genre Classification**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- 🎙️ **Multiple Input Methods**: Upload audio files (.wav, .mp3) or record directly in browser
- 🔊 **Audio Feature Extraction**: MFCCs, Chroma, Spectral Contrast, Tonnetz, ZCR, RMS, Tempo
- 🤖 **ML Model Comparison**: Random Forest, SVM, KNN, Logistic Regression
- 📊 **Confidence Scores**: Get prediction probabilities for all genres
- 🌐 **Modern Web Interface**: Built with Next.js and Tailwind CSS
- ☁️ **Cloud Deployment**: Backend on Render, Frontend on Vercel

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Audio Input   │───▶│  Feature         │───▶│   ML Model      │
│ (.wav/.mp3)     │    │  Extraction      │    │   (SVM)         │
│                 │    │  (librosa)       │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐           ▼
│   Frontend      │◀───│    FastAPI       │    ┌─────────────────┐
│  (Next.js)      │    │    Backend       │◀───│   Predictions   │
│                 │    │                  │    │  + Confidence   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
Node.js 16+ (for frontend)
```

### 1. Clone the Repository

```bash
git clone https://github.com/aaliawadkar1211/music-genre-detection-api
cd music-genre-detection-api
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 3. Frontend Setup (Optional)

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 4. Test the API

```bash
# Test with a sample audio file
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/your/audio.wav"
```

## 📊 Model Performance

The system was trained on the GTZAN dataset with the following results:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| **SVM (RBF)** | **80.2%** | **0.81** | **0.80** | **0.80** |
| Random Forest | 76.5% | 0.77 | 0.77 | 0.76 |
| Logistic Regression | 72.3% | 0.73 | 0.72 | 0.72 |
| KNN | 69.8% | 0.70 | 0.70 | 0.69 |

## 🔧 API Reference

### `POST /predict`

Classifies the genre of an uploaded audio file.

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (audio file: .wav, .mp3)

**Response:**
```json
{
  "predicted_genre": "jazz",
  "confidence": 87.45,
  "all_probabilities": {
    "blues": 0.02,
    "classical": 0.03,
    "country": 0.05,
    "disco": 0.01,
    "hiphop": 0.01,
    "jazz": 0.8745,
    "metal": 0.02,
    "pop": 0.01,
    "reggae": 0.003,
    "rock": 0.001
  }
}
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework for building APIs
- **scikit-learn**: Machine learning library for model training and inference
- **librosa**: Audio analysis library for feature extraction
- **NumPy & Pandas**: Data manipulation and numerical computations

### Frontend
- **Next.js**: React framework for building the web interface
- **Tailwind CSS**: Utility-first CSS framework for styling

### Deployment
- **Render**: Backend API hosting
- **Vercel**: Frontend hosting
- **GitHub**: Version control and CI/CD

## 📈 Development Process

1. **Data Exploration**: Analyzed the GTZAN dataset (1000 songs, 10 genres)
2. **Feature Engineering**: Extracted meaningful audio features using librosa
3. **Model Training**: Compared multiple ML algorithms and selected SVM
4. **API Development**: Built RESTful API with FastAPI
5. **Frontend Development**: Created responsive web interface
6. **Testing & Validation**: Ensured model reliability and API robustness
7. **Deployment**: Deployed to cloud platforms with CI/CD


## 🔮 Future Improvements

- [ ] **Deep Learning Models**: Implement CNN for spectrograms to improve accuracy
- [ ] **Real-time Classification**: Add live audio stream processing
- [ ] **More Genres**: Expand to include additional music genres
- [ ] **Batch Processing**: Support multiple file uploads
- [ ] **Audio Visualization**: Add spectrogram and waveform displays


## 👨‍💻 Author

**Aalia Wadkar**


---

⭐ If you found this project helpful, please consider giving it a star!

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [librosa Documentation](https://librosa.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [GTZAN Dataset]([http://marsyas.info/downloads/datasets.html](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification))
