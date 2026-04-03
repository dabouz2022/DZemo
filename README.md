# EMODZ: Professional Sentiment Analysis AI

A state-of-the-art, **100% Local AI engine** specifically trained and optimized for the unique linguistic landscape of Algeria. 

**[🌐 Access EMODZ Dashboard (DZMO)](http://209.38.249.154:8501)**

---

## 🧠 The AI-Centric Advantage

### 1. Proprietary DziriBERT Model
Unlike generic sentiment tools, EMODZ is built around a custom-trained **DziriBERT** model. 
- **Linguistic Mastery**: Native understanding of **Darija**, **Arabic**, and **French** in a social media context.
- **11-Emotion Nuance**: Beyond simple positive/negative. EMODZ detects Humor, Sarcasm, Urgency, Frustration, and more.
- **In-House Intelligence**: The model weights are included locally (`saved_model/`), ensuring the AI remains your proprietary asset.

### 2. 100% API-Free Architecture
EMODZ does **not** use external AI APIs (like OpenAI or Claude) or social media APIs.
- **Local Summarization**: Executive briefs are generated using a locally-hosted **Gemma 3:4b** via Ollama.
- **Zero Data Leakage**: Your data and insights never leave your hardware. This is essential for competitive intelligence and privacy.
- **Cost Efficient**: No recurring API bills or token costs. The intelligence is yours, 24/7.

### 3. Integrated AI Scraping
A high-performance extraction engine designed to feed the AI directly.
- **High-Yield Capture**: Bypasses login barriers and deep-expands threads to provide the AI with the highest-quality raw data.
- **Anonymous Stream**: Strips usernames and avatars to focus strictly on pure sentiment intelligence.

---

## 🛠️ Deployment (Zero-Touch)

**1. Clone the repository**
Ensure [Git LFS](https://git-lfs.com/) is installed to download the proprietary AI weights.
```bash
git lfs install
git clone <your-repository-url>
cd EMODZ
```

**2. Launch**
Double-click **`Start_EMODZ.bat`**. The system will automatically provision the local AI environment and launch the dashboard.

---

## 🧪 Tech Stack
- **Proprietary AI**: DziriBERT (Emotion Classification).
- **Local Summary**: Gemma 3:4b (via Ollama).
- **Engine**: Playwright (Non-API Scraping) + Litellm.
- **Interface**: SaaS-ready Streamlit + Plotly.

## 🛡️ Privacy Commitment
EMODZ is designed for **total data sovereignty**. Every byte of social media data, every sentiment score, and every executive brief is processed on your local machine or secure server.
