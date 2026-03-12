# 🧠 Sentiment Analysis Chatbot

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gensim-Word2Vec-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accuracy-86.28%25-brightgreen?style=for-the-badge"/>
</p>

<p align="center">
  A production-ready NLP chatbot that classifies user-input text into <strong>Positive</strong>, <strong>Neutral</strong>, or <strong>Negative</strong> sentiment using a trained <strong>LSTM neural network</strong> powered by <strong>Word2Vec embeddings</strong>.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Text Preprocessing Pipeline](#-text-preprocessing-pipeline)
- [Model Details](#-model-details)
- [Training Results](#-training-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project implements an end-to-end **Sentiment Analysis Chatbot** that predicts the emotional tone of any given text input. It leverages state-of-the-art NLP preprocessing techniques combined with a deep learning **LSTM (Long Short-Term Memory)** model trained on over **84,000 labeled text samples**.

The system categorizes input text into one of three sentiment classes:

| Label | Sentiment | Numeric Value |
|-------|-----------|---------------|
| 😊 Positive | Expresses approval, happiness, or praise | `1` |
| 😐 Neutral  | No strong emotional leaning | `0` |
| 😠 Negative | Expresses disapproval, sadness, or criticism | `-1` |

---

## 🎥 Demo

> **Example Predictions:**

```
Input  : "happy be happy"
Output : ✅ Positive

Input  : "Hi, Brother"
Output : 😐 Neutral

Input  : "bad to bad"
Output : ❌ Negative
```

---

## ✨ Features

- **Comprehensive Text Cleaning** — Removes noise such as digits, URLs, mentions, hashtags, HTML tags, and special characters
- **Contraction Expansion** — Converts `"don't"` → `"do not"` for consistent tokenization
- **Elongation Normalization** — Reduces `"soooo"` → `"soo"` to handle informal writing styles
- **Negation Handling** — Preserves negation context (e.g., `"not good"`) for accurate sentiment signals
- **POS Filtering** — Retains only semantically significant words: nouns, verbs, adjectives, adverbs
- **Language Filtering** — Removes non-English words using `langdetect`
- **Word2Vec Embeddings** — Trains 150-dimensional semantic word vectors on the full corpus
- **LSTM Sequence Modeling** — Captures long-range dependencies in text for accurate predictions
- **Interactive UI** — Real-time predictions via a clean Streamlit web interface

---

## 🏗️ Architecture

```
Raw Text Input
      │
      ▼
┌─────────────────────────────┐
│     Text Preprocessing      │
│  • Remove noise & HTML      │
│  • Expand contractions      │
│  • Normalize elongations    │
│  • Handle negations         │
│  • Lemmatize + POS filter   │
│  • Remove non-English words │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Word2Vec Embedding Layer  │
│   vocab_dim = 150           │
│   window_size = 7           │
│   min_count = 15            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│    Sequence Padding         │
│    maxlen = 100             │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│       LSTM Network          │
│  Embedding(6836 → 150)      │
│  LSTM(units=50, tanh)       │
│  Dropout(0.5)               │
│  Dense(3, softmax)          │
└────────────┬────────────────┘
             │
             ▼
    Sentiment Prediction
  (Positive / Neutral / Negative)
```

---

## 📁 Project Structure

```
mohamedsahadm786-sentiment-analysis-chatbot/
│
├── 📓 SENTIMENT ANALYSIS.ipynb   # Main Jupyter notebook (training + evaluation)
│
├── 🤖 lstm.json                  # Saved LSTM model architecture (JSON)
├── ⚖️  lstm.weights.h5            # Trained LSTM model weights
├── 📦 Word2vec_model.pkl         # Trained Word2Vec embedding model
│
├── 📊 pos.csv                    # Positive sentiment training data (~28,028 samples)
├── 📊 neg.csv                    # Negative sentiment training data (~28,926 samples)
├── 📊 neutral.csv                # Neutral sentiment training data (~27,673 samples)
│
├── 📄 Technologies Used.txt      # Summary of libraries and tools used
└── 📖 README.md                  # Project documentation (this file)
```

---

## 📊 Dataset

The model is trained on **84,627 labeled text samples** across three balanced CSV files:

| File | Sentiment | Samples |
|------|-----------|---------|
| `pos.csv` | Positive | 28,028 |
| `neg.csv` | Negative | 28,926 |
| `neutral.csv` | Neutral | 27,673 |
| **Total** | **—** | **84,627** |

> The dataset primarily consists of **movie reviews** and social media-style text, covering a diverse range of writing styles including formal critique, casual speech, and informal internet language.

---

## 🧹 Text Preprocessing Pipeline

Each input text goes through an 8-step cleaning pipeline before being fed into the model:

```
Step 1 → Remove digits
Step 2 → Remove URLs, @mentions, #hashtags
Step 3 → Strip HTML tags
Step 4 → Remove special characters & punctuation
Step 5 → Expand contractions  (e.g., "isn't" → "is not")
Step 6 → Normalize elongated words  (e.g., "loooove" → "loove")
Step 7 → Lemmatize + POS filter + Handle negations  (via SpaCy)
Step 8 → Remove non-English words  (via LangDetect)
```

**Before cleaning:**
```
"Henry Selick's first movie since 2009's Coraline. His fifth stop-motion masterpiece."
```

**After cleaning:**
```
['henry', 'selick', 'movie', 'coraline', 'fifth', 'stopmotion', 'masterpiece']
```

---

## 🤖 Model Details

### Word2Vec Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `vector_size` | 150 | Embedding dimensionality |
| `window` | 7 | Context window size |
| `min_count` | 15 | Minimum word frequency |
| `epochs` | 15 | Training iterations |
| `workers` | CPU count | Parallel processing |

### LSTM Network Configuration

| Layer | Output Shape | Parameters |
|-------|-------------|------------|
| `InputLayer` | `(None, 100)` | — |
| `Embedding` | `(None, 100, 150)` | vocab × 150 |
| `LSTM` | `(None, 50)` | 50 units, tanh activation |
| `Dropout` | `(None, 50)` | rate = 0.5 |
| `Dense` | `(None, 3)` | softmax activation |
| `Activation` | `(None, 3)` | softmax |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr = 0.001) |
| Loss Function | Categorical Crossentropy |
| Batch Size | 32 |
| Epochs | 4 |
| Train / Test Split | 80% / 20% |
| Sequence Max Length | 100 tokens |

---

## 📈 Training Results

| Epoch | Accuracy | Loss |
|-------|----------|------|
| 1 | 72.71% | 0.8325 |
| 2 | 84.87% | 0.7063 |
| 3 | 87.92% | 0.6738 |
| 4 | 89.91% | 0.6533 |

**Final Test Evaluation:**

| Metric | Score |
|--------|-------|
| ✅ Test Accuracy | **86.28%** |
| 📉 Test Loss | 0.6872 |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/mohamedsahadm786/sentiment-analysis-chatbot.git
cd sentiment-analysis-chatbot
```

### 2. Install Dependencies

```bash
pip install pandas numpy spacy keras tensorflow gensim langdetect contractions streamlit scikit-learn
```

### 3. Download the SpaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Usage

### Run Inference (Python)

```python
from lstm_predict import lstm_predict

lstm_predict("This movie was absolutely amazing!")
# Output: Positive

lstm_predict("The film had a few good moments.")
# Output: Neutral

lstm_predict("Terrible acting and a boring plot.")
# Output: Negative
```

### Launch the Streamlit App

```bash
streamlit run app.py
```

Then open your browser and navigate to `http://localhost:8501`.

---

## 🛠️ Technologies Used

| Category | Tool / Library | Purpose |
|----------|---------------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **Data Processing** | Pandas, NumPy | Data manipulation & arrays |
| **NLP Preprocessing** | SpaCy (`en_core_web_sm`) | Tokenization, lemmatization, POS tagging |
| **Contraction Handling** | `contractions` | Expanding shortened forms |
| **Language Detection** | `langdetect` | Filtering non-English words |
| **Word Embeddings** | Gensim Word2Vec | Semantic word vector training |
| **Deep Learning** | Keras + TensorFlow | LSTM model definition & training |
| **Model Serialization** | `.json` + `.h5` | Architecture and weight storage |
| **ML Utilities** | scikit-learn | Train/test split, accuracy score |
| **UI / Interface** | Streamlit | Interactive web chatbot |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedsahadm786">mohamedsahadm786</a>
</p>
