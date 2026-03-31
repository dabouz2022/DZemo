import sys
import streamlit as st
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import json
import logging
import pandas as pd
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from crawl4ai import AsyncWebCrawler
import litellm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi.algorithm import get_display

# Configure Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title="DzEmotion Dashboard", page_icon="🇩🇿", layout="wide")

EMOTION_CLASSES = [
    'Satisfaction', 'Frustration', 'Urgency', 'Sarcasm', 'Inquiry',
    'Disappointment', 'Gratitude', 'Anger', 'Confusion',
    'Suggestion / Feedback', 'Humor / Joking'
]

HIGH_PRIORITY_EMOTIONS = ['Urgency', 'Anger', 'Sarcasm']

@st.cache_resource
def load_local_model():
    model_path = "./saved_model"
    logger.info(f"Loading tokenizer and model from {model_path}...")
    tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(model_path, local_files_only=True, num_labels=11)
    model.eval()
    logger.info("Model and tokenizer loaded successfully.")
    return tokenizer, model

def predict_emotion(text, tokenizer, model):
    if not text:
        return "Unknown"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    if 0 <= predicted_class_id < len(EMOTION_CLASSES):
        return EMOTION_CLASSES[predicted_class_id]
    return "Unknown"

async def scrape_and_extract_comments(url):
    logger.info(f"Starting scrape_and_extract_comments for URL: {url}")

    # Ensure playwright binaries are present automatically
    try:
        import subprocess
        subprocess.run(["playwright", "install", "chromium"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        logger.warning(f"Auto playwright install skipped or failed: {e}")

    # Custom JavaScript to handle unpredictable Facebook UI (delayed popups, comment loading)
    js_interaction = """
    let attempts = 0;
    let clickInterval = setInterval(() => {
        const closeBtn = document.querySelector('div[aria-label="Close"]') || document.querySelector('div[aria-label="Fermer"]') || document.querySelector('div[aria-label="\u0625\u063a\u0644\u0627\u0642"]');
        if (closeBtn) closeBtn.click();

        const dialogs = document.querySelectorAll('div[role="dialog"]');
        dialogs.forEach(d => d.remove());

        document.querySelectorAll('[aria-hidden="true"]').forEach(el => el.removeAttribute('aria-hidden'));
        document.dispatchEvent(new KeyboardEvent('keydown', {'key': 'Escape'}));

        const buttons = Array.from(document.querySelectorAll('div[role="button"]'));
        const commentBtn = buttons.find(el =>
            el.textContent.includes('View more comments') ||
            el.textContent.includes('Afficher plus') ||
            el.textContent.includes('\u0639\u0631\u0636 \u0627\u0644\u0645\u0632\u064a\u062f')
        );
        if (commentBtn) commentBtn.click();

        attempts++;
        if (attempts > 10) clearInterval(clickInterval);
    }, 500);
    """

    logger.info("Starting AsyncWebCrawler...")
    async with AsyncWebCrawler(headless=False) as crawler:
        result = await crawler.arun(
            url=url,
            magic_mode=False,
            bypass_cache=True,
            delay_before_return_html=5.0,
            js_code=js_interaction,
            max_scroll_steps=10,
            scroll_delay=3.0,
            word_count_threshold=0
        )

    logger.info(f"Crawler finished. Success: {result.success}")
    md_len = len(result.markdown) if result.markdown else 0
    logger.info(f"Raw Markdown Length: {md_len} chars.")

    if md_len < 200:
        logger.warning(f"Very short markdown! Page may be gated. Preview: {(result.markdown or '')[:200]}")
        return None

    # --- DIRECT OLLAMA CALL: bypass LLMExtractionStrategy's silent failures ---
    logger.info("Sending markdown to Ollama (gemma3:4b) for comment extraction...")
    # Dynamically find where comments start in the markdown.
    # Facebook always places a "Most relevant" / "Like · Comment · Share" divider before comments.
    md = result.markdown
    comment_section_start = -1
    for marker in ["Most relevant", "Most Relevant", "Top comments", "Like · Comment · Share", "Like\nComment\nShare", "\u0627\u062a\u0631\u0643 \u062a\u0639\u0644\u064a\u0642\u0627\u064b", "\u062a\u0639\u0644\u064a\u0642\u0627\u062a"]:
        idx = md.find(marker)
        if idx != -1:
            comment_section_start = idx
            logger.info(f"Found comments section at char {idx} using marker: '{marker}'")
            break

    if comment_section_start == -1:
        logger.warning("Could not find comment section marker. Sending second half of markdown as fallback.")
        comment_section_start = len(md) // 2

    page_text = md[comment_section_start:]
    logger.info(f"Comments section is {len(page_text)} chars. Processing in chunks...")

    CHUNK_SIZE = 3000
    all_comments = []
    chunks = [page_text[i:i+CHUNK_SIZE] for i in range(0, len(page_text), CHUNK_SIZE)]
    logger.info(f"Splitting into {len(chunks)} chunks of ~{CHUNK_SIZE} chars each.")

    for i, chunk in enumerate(chunks):
        logger.info(f"Calling Ollama on chunk {i+1}/{len(chunks)}...")
        prompt = (
            "You are a data extraction assistant. "
            "From the following text scraped from a Facebook post's comment section, extract ALL user comments. "
            "Return ONLY a valid JSON array of objects, each with three keys: 'user' (string), 'text' (string, the comment text), and 'timestamp' (string, e.g. '5w', '2d', '1h'. If no timestamp is visible, use 'Unknown'). "
            "Ignore navigation links, 'Like', 'Reply', and author badges. "
            "Keep the original Arabic/Darija/French/Emoji text exactly as-is. "
            "If no comments are found in this chunk, return an empty array []. "
            "Do not add any explanation before or after the JSON.\n\n"
            f"--- COMMENTS CHUNK {i+1} ---\n{chunk}"
        )
        try:
            response = litellm.completion(
                model="ollama/gemma3:4b",
                messages=[{"role": "user", "content": prompt}],
                api_base="http://localhost:11434",
                num_ctx=2048,
            )
            if not response.choices or not response.choices[0].message.content:
                logger.warning(f"Chunk {i+1}: Ollama returned empty response, skipping.")
                continue
            raw_text = response.choices[0].message.content
            logger.info(f"Chunk {i+1}: Ollama responded with {len(raw_text)} chars.")

            start_idx = raw_text.find('[')
            end_idx = raw_text.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                logger.warning(f"Chunk {i+1}: No JSON array found. Raw: {raw_text[:300]}")
                continue
            chunk_comments = json.loads(raw_text[start_idx:end_idx])
            logger.info(f"Chunk {i+1}: Extracted {len(chunk_comments)} comments.")
            all_comments.extend(chunk_comments)
        except Exception as e:
            logger.error(f"Chunk {i+1}: Ollama call failed: {e}")
            continue

    logger.info(f"Total comments extracted across all chunks: {len(all_comments)}")
    if not all_comments:
        return None
    return json.dumps(all_comments)


EMOTION_META = {
    'Satisfaction':        {'emoji': '😊', 'color': '#22c55e'},
    'Frustration':         {'emoji': '😤', 'color': '#f97316'},
    'Urgency':             {'emoji': '🚨', 'color': '#ef4444'},
    'Sarcasm':             {'emoji': '😏', 'color': '#a855f7'},
    'Inquiry':             {'emoji': '🤔', 'color': '#3b82f6'},
    'Disappointment':      {'emoji': '😞', 'color': '#64748b'},
    'Gratitude':           {'emoji': '🙏', 'color': '#10b981'},
    'Anger':               {'emoji': '😡', 'color': '#dc2626'},
    'Confusion':           {'emoji': '😕', 'color': '#eab308'},
    'Suggestion / Feedback': {'emoji': '💡', 'color': '#06b6d4'},
    'Humor / Joking':      {'emoji': '😂', 'color': '#f59e0b'},
}

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=Noto+Sans+Arabic:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-dark: #0a0a0a;
    --sidebar-bg: #111111;
    --card-bg: rgba(255, 255, 255, 0.03);
    --border-color: rgba(255, 255, 255, 0.08);
    --accent-color: #ffffff;
    --text-primary: #f0f0f0;
    --text-secondary: #9ca3af;
    --primary-font: 'Inter', 'Noto Sans Arabic', sans-serif;
    --heading-font: 'Outfit', 'Noto Sans Arabic', sans-serif;
}

/* Global Reset & Typography */
html, body, .stApp, .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, li, label, .stSidebar, .stButton button, .stTextInput input, .stTextArea textarea, .stSelectbox {
    font-family: var(--primary-font) !important;
}

/* Allow Material Icons to render correctly */
.material-symbols-rounded, .material-icons, span[class*="icon"], span[class*="material"], [data-testid="collapsedControl"] span {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

.stApp {
    background: var(--bg-dark);
    color: var(--text-primary);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg);
    border-right: 1px solid var(--border-color);
}
[data-testid="stSidebar"] section {
    padding-top: 2rem;
}

/* Content Width & Alignment */
.block-container {
    max-width: 900px !important;
    padding-top: 4rem !important;
}

/* Sophisticated Header */
.local-badge {
    display: inline-block;
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    color: #4ade80;
    padding: 2px 10px;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}
.top-kicker {
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.3em;
    font-weight: 600;
    margin-bottom: 0.5rem;
    text-align: center;
}
.gradient-title {
    font-family: var(--heading-font) !important;
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 0.75rem;
    text-align: center;
    color: #ffffff;
}
.subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    text-align: center;
    max-width: 600px;
    margin: 0 auto 3rem;
    line-height: 1.5;
}

/* Tabs UI */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 2rem;
}
.stTabs [data-baseweb="tab"] {
    height: 45px;
    background-color: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0 1.5rem;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
}

/* Refined Form Controls */
.stTextInput input, .stTextArea textarea {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    color: #ffffff !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
}
.stTextInput label, .stTextArea label {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #ffffff !important;
    margin-bottom: 8px !important;
}

/* Premium Primary Button */
.stButton button[kind="primary"] {
    background: #ffffff !important;
    color: #000000 !important;
    border-radius: 6px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    width: 100%;
    margin-top: 1rem;
    border: none !important;
    transition: transform 0.2s;
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-1px);
    background-color: #f3f4f6 !important;
}

/* Sidebar Info Cards */
.info-box {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-color);
    padding: 1.25rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}
.info-title {
    font-size: 0.7rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #ffffff;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.info-content {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Analysis KPIs */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
    margin: 3rem 0;
}
.kpi-card {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid var(--border-color);
    padding: 1.5rem;
    border-radius: 8px;
    text-align: center;
}
.kpi-value {
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
}
.kpi-label {
    font-size: 0.65rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary) !important;
    margin-top: 0.5rem;
    font-weight: 600 !important;
}

/* Section Dividers */
.section-title {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #6b7280 !important;
    margin: 4rem 0 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}
</style>
"""


def run_emotion_analysis(comments_data, tokenizer, model):
    results = []
    for idx, item in enumerate(comments_data):
        if isinstance(item, dict):
            text = item.get("text", "")
            timestamp = item.get("timestamp", "Unknown")
        else:
            text = str(item)
            timestamp = "Unknown"
            
        if not text or not text.strip():
            continue
        emotion = predict_emotion(text, tokenizer, model)
        is_high_priority = emotion in HIGH_PRIORITY_EMOTIONS
        logger.info(f"Comment {idx+1} | Emotion: {emotion} | Text: {text[:60]}")
        results.append({
            "Timestamp": timestamp,
            "Comment": text,
            "Emotion": emotion,
            "High Priority": "🚨 YES" if is_high_priority else "NO"
        })
    return results

def generate_executive_brief(df):
    try:
        top_issues = df[df["High Priority"] == "🚨 YES"]["Comment"].tolist()[:10]
        general = df["Comment"].tolist()[:15]
        
        prompt = (
            "You are an expert Social Media Sentiment Analyst. I have collected comments from Algerian users on a recent post. "
            "Write a strict 3-sentence Executive Brief summarizing the overall sentiment. "
            "Mention the main source of frustration or anger (if any) and the general vibe (if people are satisfied). "
            "Keep it highly professional, short, and to the point.\n\n"
            f"High Priority Comments (Anger/Frustration): {top_issues}\n"
            f"General Comments: {general}"
        )
        response = litellm.completion(
            model="ollama/gemma3:4b",
            messages=[{"role": "user", "content": prompt}],
            api_base="http://localhost:11434",
            num_ctx=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error generating AI Brief: {e}")
        return "⚠️ Could not generate AI brief. Please ensure Ollama is running."


def render_results(results):
    if not results:
        st.warning("No comments to display.")
        return

    df = pd.DataFrame(results)
    logger.info(f"Rendering {len(df)} results.")

    high_prio_count = (df["High Priority"] == "🚨 YES").sum()
    top_emotion = df["Emotion"].value_counts().index[0] if len(df) > 0 else "N/A"
    top_meta = EMOTION_META.get(top_emotion, {'emoji': '❓', 'color': '#94a3b8'})

    # ── KPI Cards ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-label">Total Comments</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value" style="color:#ffffff;">{high_prio_count}</div>
            <div class="kpi-label">High Priority</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{top_meta['emoji']}</div>
            <div class="kpi-label">Top: {top_emotion}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-value">{df['Emotion'].nunique()}</div>
            <div class="kpi-label">Emotions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Executive Brief ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">AI Executive Brief</div>', unsafe_allow_html=True)
    if st.button("Generate Summary", type="primary"):
        with st.spinner("Analyzing feed..."):
            brief = generate_executive_brief(df)
            st.markdown(f"""
            <div style="background:#181818; border:1px solid #333; padding:32px 40px; margin-bottom:2rem;">
                <div style="color:#d0d0d0; font-size:0.95rem; line-height:1.8; font-family:serif; font-style:italic;">"{brief}"</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    emotion_counts = df['Emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    colors = [EMOTION_META.get(e, {}).get('color', '#6366f1') for e in emotion_counts['Emotion']]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<div class="section-title">Distribution</div>', unsafe_allow_html=True)
        fig_donut = px.pie(
            emotion_counts, names='Emotion', values='Count',
            hole=0.65, color='Emotion',
            color_discrete_map={e: EMOTION_META.get(e, {}).get('color', '#6366f1') for e in emotion_counts['Emotion']}
        )
        fig_donut.update_traces(
            textposition='inside', 
            textinfo='label+percent',
            insidetextorientation='radial',
            marker=dict(line=dict(color='#121212', width=4)),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<extra></extra>"
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff', 
            font_family='Montserrat',
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=340,
            hoverlabel=dict(bgcolor="#ffffff", font_color="#000000", font_size=12, font_family="Montserrat")
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Volume</div>', unsafe_allow_html=True)
        fig_bar = px.bar(emotion_counts, x='Count', y='Emotion', orientation='h',
                         color='Emotion',
                         color_discrete_map={e: EMOTION_META.get(e, {}).get('color', '#6366f1') for e in emotion_counts['Emotion']})
        
        fig_bar.update_traces(
            marker_line_width=0, 
            opacity=1.0,
            hovertemplate="<b>%{y}</b>: %{x}<extra></extra>"
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#aaaaaa', 
            font_family='Montserrat',
            showlegend=False,
            yaxis=dict(autorange="reversed", title="", showgrid=False, tickfont=dict(size=10)),
            xaxis=dict(title="", showgrid=True, gridcolor='#333333', tickfont=dict(size=10)),
            margin=dict(t=10, b=10, l=10, r=20),
            height=340,
            hoverlabel=dict(bgcolor="#ffffff", font_color="#000000", font_size=12, font_family="Montserrat")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Comment Cards ──
    st.markdown('<div class="section-title">Feed</div>', unsafe_allow_html=True)
    for _, row in df.iterrows():
        emotion = row["Emotion"]
        meta = EMOTION_META.get(emotion, {'emoji': '❓', 'color': '#ffffff'})
        priority_badge = f' <span style="color:#ffffff;font-size:0.6rem;font-weight:700;letter-spacing:0.2em;border:1px solid #ffffff;padding:2px 8px;margin-left:8px;">HIGH PRIORITY</span>' if row["High Priority"] == "🚨 YES" else ''
        text = str(row['Comment']).replace('<', '&lt;').replace('>', '&gt;')
        st.markdown(f"""
        <div style="background:#181818; border:1px solid #333; padding:24px; margin-bottom:16px;">
            <div style="font-size:0.9rem; color:#eeeeee; line-height:1.6; margin-bottom:20px; direction:auto; font-weight:400;">{text}</div>
            <div style="display:flex; align-items:center;">
                <span style="font-size:0.65rem; font-weight:700; color:#888; text-transform:uppercase; letter-spacing:0.15em;">
                    {meta['emoji']} {emotion}
                </span>{priority_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Download ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Results as CSV", csv_data, "dzemotions_results.csv", "text/csv")


def main():
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:16px 0 24px;'>
            <div style='font-size:2.5rem;margin-bottom:8px;'>🇩🇿</div>
            <div style='font-size:1.4rem;font-weight:800;color:#ffffff;font-family:"Outfit";letter-spacing:-0.5px;'>DzEmotion</div>
            <div style='font-size:0.7rem;color:#888;text-transform:uppercase;letter-spacing:0.1em;font-weight:600;'>Algerian Sentiment Hub</div>
        </div>
        
        <div class="info-box">
            <div class="info-title">🛡️ Privacy & Security</div>
            <div class="info-content">
                <b>Local-First Architecture:</b> This application leverages a self-hosted AI stack. 
                All data, including comments and sentiment logs, is processed within a secure 
                local environment. No data is shared with external third-party clouds or AI APIs.
            </div>
        </div>

        <div class="info-box">
            <div class="info-title">📊 Description</div>
            <div class="info-content">
                Advanced social intelligence platform tailored for the Algerian digital landscape. 
                Analyze public sentiment across digital platforms using <b>DziriBERT</b> for nuanced emotion detection 
                in Darija, Arabic, and French.
            </div>
        </div>

        <hr style='border-color:rgba(255,255,255,0.05);margin:24px 0;'>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='font-size:0.75rem; color:#aaa;'>
            <p><b>Core Stack:</b></p>
            <ul style='list-style-type:none; padding:0;'>
                <li>🧠 DziriBERT (Local)</li>
                <li>🤖 Gemma 3:4b (Local)</li>
                <li>🕸️ AsyncWebCrawler</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div style='text-align:center;'>
        <div class='local-badge'>● Local Mode Active</div>
    </div>
    <div class='top-kicker'>Social Content Analytics</div>
    <div class='gradient-title'>DZEMOTION</div>
    <div class='subtitle'>The definitive sentiment analysis engine for Algerian social media</div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading offline AI engine..."):
        try:
            tokenizer, model = load_local_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            st.error(f"Error loading local model from ./saved_model: {e}")
            return

    tab1, tab2 = st.tabs(["🌐  Scrape from URL", "📋  Manual Input / CSV Upload"])

    # ── TAB 1: URL scraping ───────────────────────────────────────────────
    with tab1:
        st.markdown("Enter a **public post permalink** to auto-scrape and analyze comments.")
        url_input = st.text_input("Post URL:", key="url_input",
                                   placeholder="https://www.facebook.com/permalink.php?story_fbid=...")
        start_analysis = st.button("🔍 Scrape & Analyse", type="primary", key="btn_url")

        if start_analysis and url_input:
            with st.status("Scraping page and analysing sentiments...", expanded=True) as status:
                try:
                    st.write("Launching browser and bypassing bot protection...")
                    extracted_json_str = asyncio.run(scrape_and_extract_comments(url_input))
                    if not extracted_json_str:
                        status.update(label="No content extracted.", state="error")
                        st.warning("No comments could be extracted. Make sure Ollama is running and the URL is a public Facebook post permalink.")
                        return
                    clean = extracted_json_str.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    comments_list = json.loads(clean)
                    if not isinstance(comments_list, list) or len(comments_list) == 0:
                        status.update(label="No comments found.", state="error")
                        st.warning("Page was found but no comments were identified.")
                        return
                    st.write(f"Running DziriBERT on {len(comments_list)} comments...")
                    data_payload = [{"text": item.get("text", str(item)), "timestamp": item.get("timestamp", "Unknown")} for item in comments_list]
                    results = run_emotion_analysis(data_payload, tokenizer, model)
                    st.session_state['results'] = results
                    status.update(label=f"Done! Analysed {len(results)} comments.", state="complete", expanded=False)
                except Exception as e:
                    logger.exception(f"Unexpected error: {e}")
                    status.update(label=f"Analysis failed. Is Ollama running?", state="error")
                    st.error(f"Error during extraction or classification: {e}")
                    st.info("Make sure you have Ollama installed and running with `ollama run gemma3:4b`")

    # ── TAB 2: Manual + CSV ───────────────────────────────────────────────
    with tab2:
        st.markdown("Analyse comments **without scraping** — paste them or upload a CSV.")
        input_method = st.radio("Input method:", ["✏️ Type / Paste Comments", "📂 Upload CSV File"], horizontal=True)

        comments_texts = []
        run_analysis = False

        if input_method == "✏️ Type / Paste Comments":
            manual_text = st.text_area(
                "One comment per line:",
                height=220,
                placeholder="طبيعي مكش تجيبه؟\nبتوفيق انشاء الله\nC'est vraiment bien!"
            )
            if st.button("🔍 Analyse Comments", type="primary", key="btn_manual"):
                comments_texts = [{"text": l.strip(), "timestamp": "Unknown"} for l in manual_text.splitlines() if l.strip()]
                run_analysis = True

        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file:
                try:
                    df_csv = pd.read_csv(uploaded_file)
                    st.success(f"Loaded {len(df_csv)} rows — columns: {list(df_csv.columns)}")
                    col_name = None
                    for c in ["comment", "comments", "text", "Comment", "Comments", "Text", "تعليق", "تعليقات"]:
                        if c in df_csv.columns:
                            col_name = c
                            break
                    if not col_name:
                        col_name = st.selectbox("Which column contains the comments?", df_csv.columns.tolist())
                    if col_name:
                        st.info(f"Using column: **{col_name}**")
                        st.dataframe(df_csv[[col_name]].head(8), use_container_width=True)
                        
                        date_col = None
                        for c in ["date", "time", "timestamp", "Date", "Time", "Timestamp", "تاريخ"]:
                            if c in df_csv.columns:
                                date_col = c
                                st.info(f"Auto-detected time column: **{date_col}**")
                                break
                                
                        if st.button("🔍 Analyse CSV", type="primary", key="btn_csv"):
                            comments_texts = []
                            for _, row in df_csv.iterrows():
                                if pd.notnull(row[col_name]):
                                    comments_texts.append({
                                        "text": str(row[col_name]),
                                        "timestamp": str(row[date_col]) if date_col and pd.notnull(row[date_col]) else "Unknown"
                                    })
                            run_analysis = True
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        if run_analysis and comments_texts:
            with st.spinner(f"Predicting emotions for {len(comments_texts)} comments..."):
                results = run_emotion_analysis(comments_texts, tokenizer, model)
                st.session_state['results'] = results
            st.success(f"✅ Analysed {len(results)} comments.")
        elif run_analysis and not comments_texts:
            st.warning("No comments found. Please paste some text or upload a valid CSV.")

    if 'results' in st.session_state and st.session_state['results']:
        render_results(st.session_state['results'])

if __name__ == "__main__":
    main()
