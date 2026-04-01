# EMODZ Dashboard

The definitive sentiment analysis engine for Algerian social media (Darija, Arabic, French) that focuses on privacy and local processing and data security.

## 🚀 Quick Start (100% Out-Of-The-Box)

The easiest way to run the application is to use the provided Windows batch file. 
It requires absolutely **zero manual configuration**.

**1. Clone the repository**
Make sure you have [Git LFS](https://git-lfs.com/) installed *before* cloning so that the AI model weights (`saved_model/*.safetensors`) download correctly:
```bash
# Install Git LFS once
git lfs install
# Clone repository
git clone <your-repository-url>
cd EMODZ
```

**2. Run the Auto-Installer Launcher**
Simply double-click the `Start_DzEmotion.bat` file located in the project folder!

Behind the scenes, the script will automatically:
- Check if Python is installed (and download it if it's not).
- Check if Ollama is installed (and download it if it's not).
- Pull the necessary 2.5GB Gemma 3 AI model offline via Ollama.
- Create an isolated Python Virtual Environment safely.
- Install all Pip dependencies (`requirements.txt`).
- Download the embedded auto-driving Chromium browser for web scraping (Playwright).
- Launch the main UI.

Next time you open it, everything will be instant!

## Features
- **DziriBERT Integration**: Fine-tuned localized emotion classification for 11 distinct Algerian emotional scopes.
- **Local-First Scraping**: No external APIs used. We automatically scrape URLs directly and extract locally.
- **Playwright Auto-Setup**: Chromium dependencies automatically load on your first scrape via Playwright.

## Troubleshooting
- **Missing or Corrupted Model**: If `./saved_model` gives errors, verify you downloaded it fully (using Git LFS) and not just pointer text files.
- **Ollama Missing**: If the AI extraction engine fails, make sure Ollama is actually running in your Windows taskbar.
