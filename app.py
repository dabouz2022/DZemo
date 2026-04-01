import sys
import streamlit as st
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import os
import re
import json
import html as html_lib
import logging
import random
import time
import shutil
import socket
import subprocess
from pathlib import Path
from contextlib import nullcontext
from urllib.parse import urlparse, urlunparse
import pandas as pd
import plotly.express as px
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from playwright.async_api import async_playwright
import litellm
try:
    from pyvirtualdisplay import Display
except Exception:
    Display = None

# Configure Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

RUNTIME_DIR = Path(".runtime")
CHROME_RUNTIME_DIR = RUNTIME_DIR / "chrome"


def ensure_runtime_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)


def find_real_chrome_executable():
    candidates = []
    if sys.platform == "win32":
        candidates.extend([
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
            Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
        ])
    elif sys.platform == "darwin":
        candidates.extend([
            Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
            Path.home() / "Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ])
    elif sys.platform.startswith("linux"):
        for binary in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"]:
            located = shutil.which(binary)
            if located:
                candidates.append(Path(located))
        
        # Check Playwright's cache directory (common in CI/CD and server environments)
        # Search patterns: ~/.cache/ms-playwright/chromium-*/chrome-linux*/chrome
        playwright_cache = Path.home() / ".cache" / "ms-playwright"
        if playwright_cache.exists():
            try:
                for chrom_dir in playwright_cache.glob("chromium-*"):
                    for platform_dir in chrom_dir.glob("chrome-linux*"):
                        chrome_bin = platform_dir / "chrome"
                        if chrome_bin.exists():
                            candidates.append(chrome_bin)
            except Exception:
                pass

    for candidate in candidates:
        if candidate and candidate.exists():
            return str(candidate)
    return None


def get_free_tcp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return sock.getsockname()[1]


def wait_for_cdp_ready(port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def maybe_start_virtual_display(width, height):
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY") and Display:
        try:
            display = Display(visible=False, size=(max(width, 1280), max(height, 900)))
            display.start()
            logger.info("Started virtual display (Xvfb) for headless Linux environment.")
            return display
        except Exception as e:
            logger.warning(f"Failed to start pyvirtualdisplay/Xvfb: {e}. Falling back to --headless=new mode.")
    return nullcontext()


def launch_real_chrome_for_cdp(url, profile, mobile_mode=True):
    chrome_path = find_real_chrome_executable()
    if not chrome_path:
        raise RuntimeError("Real Google Chrome executable was not found.")

    ensure_runtime_dir(CHROME_RUNTIME_DIR)
    profile_dir = ensure_runtime_dir(CHROME_RUNTIME_DIR / profile["name"])
    port = get_free_tcp_port()
    width = profile["viewport"]["width"]
    height = profile["viewport"]["height"]
    display = maybe_start_virtual_display(width, height)
    if hasattr(display, "__enter__"):
        display = display.__enter__()

    args = [
        chrome_path,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={profile_dir}",
        f"--window-size={width},{height}",
        "--lang=fr-FR",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-extensions",
        "--disable-features=Translate,OptimizationHints,MediaRouter",
        "--new-window",
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-zygote",
    ]

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        args.append("--headless=new")

    if mobile_mode:
        args.append(
            f"--user-agent={profile['user_agent']}"
        )
    args.append(url)

    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    process = subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, # Capture stderr for diagnosis if needed
        creationflags=creationflags,
    )

    if not wait_for_cdp_ready(port, timeout=65):
        try:
            # If fail, read some stderr
            err_data = process.stderr.read(1024).decode('utf-8', errors='replace')
            logger.warning(f"Chrome CDP port did not become ready. Stderr: {err_data}")
            process.terminate()
        except Exception:
            pass
        if hasattr(display, "stop"):
            display.stop()
        raise RuntimeError("Chrome CDP port did not become ready.")

    return {
        "process": process,
        "port": port,
        "display": display,
    }


def cleanup_chrome_runtime(runtime):
    if not runtime or not isinstance(runtime, dict):
        return
    process = runtime.get("process")
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
    display = runtime.get("display")
    if display and hasattr(display, "stop"):
        try:
            display.stop()
        except Exception:
            pass

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


def normalize_scraped_text(text):
    if not text:
        return ""
    text = html_lib.unescape(str(text))
    return re.sub(r"\s+", " ", text.replace("\u200b", " ").replace("\u00a0", " ").replace("\ufeff", " ")).strip()


def clean_comment_text(text):
    """Strip markdown and UI residue while preserving readable comment content."""
    text = normalize_scraped_text(text)
    text = re.sub(r'!\[([^\]]{0,40})\]\([^)]+\)', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = text.lstrip("!")
    text = re.sub(r'^[>*\-\s]+', '', text)
    return normalize_scraped_text(text)


def is_ui_residue_comment(text):
    value = normalize_scraped_text(text).lower()
    if not value:
        return True
    patterns = [
        r'^view \d+ repl',
        r'^view more',
        r'^view previous',
        r'^afficher \d+ r',
        r'^afficher plus',
        r'^charger plus',
        r'^most relevant$',
        r'^all comments$',
        r'^newest$',
        r'^comments?$',
        r'^likes?$',
        r'^replies$',
    ]
    return any(re.search(pattern, value) for pattern in patterns)


MOBILE_BROWSER_PROFILES = [
    {
        "name": "iphone_390x844",
        "viewport": {"width": 390, "height": 844},
        "user_agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 "
            "Mobile/15E148 Safari/604.1"
        ),
        "locale": "fr-FR",
    },
]

FACEBOOK_LOADING_MARKERS = [
    "loading comments",
    "loading more comments",
    "please wait",
    "be patient",
    "chargement des commentaires",
    "chargement",
    "veuillez patienter",
    "patientez",
    "kommentare werden geladen",
    "kommentare laden",
    "bitte warten",
    "einen moment bitte",
]

FACEBOOK_GATE_MARKERS = [
    "découvrez plus de contenu",
    "discover more content",
    "mehr inhalte entdecken",
    "ce contenu n'est pas disponible hors connexion",
    "this content isn't available unless you log in",
    "dieser inhalt ist nur nach der anmeldung verfügbar",
]

FACEBOOK_END_MARKERS = [
    "neueste beiträge",
]

FACEBOOK_TIMESTAMP_RE = re.compile(
    r"(?:^|\b)(?:\d+\s*(?:s|min|h|d|j|w|sem|mo|an|ans|yr|yrs)|hier|yesterday|today|aujourd'hui|gestern|heute)(?:\b|$)",
    re.IGNORECASE,
)


def to_m_facebook_url(url):
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    if "facebook.com" not in netloc and "fb.com" not in netloc:
        return url.strip()
    if netloc.endswith("fb.com"):
        netloc = "m.facebook.com"
    else:
        netloc = re.sub(r"^(?:www|m)\.", "", netloc)
        netloc = f"m.{netloc}"
    return urlunparse((parsed.scheme or "https", netloc, parsed.path, parsed.params, parsed.query, ""))


def to_www_facebook_url(url):
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    if "facebook.com" not in netloc and "fb.com" not in netloc:
        return url.strip()
    if netloc.endswith("fb.com"):
        netloc = "www.facebook.com"
    else:
        netloc = re.sub(r"^(?:m|www)\.", "", netloc)
        netloc = f"www.{netloc}"
    return urlunparse((parsed.scheme or "https", netloc, parsed.path, parsed.params, parsed.query, ""))


def normalize_name_for_match(text):
    value = normalize_scraped_text(text)
    value = re.sub(r"[.\u2026!?,:;،؛'\"`~_^*+=/\\|()\[\]{}<>-]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def looks_like_profile_name_only(text, user=""):
    value = clean_comment_text(text)
    if not value:
        return True
    normalized_value = normalize_name_for_match(value)
    normalized_user = normalize_name_for_match(user)
    
    # Only filter if it's the user's own name (common in some scrapers' outputs)
    if normalized_user and normalized_value == normalized_user:
        return True
        
    # We no longer filter by word patterns to avoid dropping valid short comments like "Good job" or "I agree".
    return False


def normalize_comment_records(records):
    cleaned = []
    seen = set()
    for item in records or []:
        if not isinstance(item, dict):
            item = {"text": str(item), "user": "", "timestamp": "Unknown"}
        user = normalize_scraped_text(item.get("user", ""))
        text = clean_comment_text(item.get("text", ""))
        timestamp = normalize_scraped_text(item.get("timestamp", "")) or "Unknown"
        if not text or is_ui_residue_comment(text):
            continue
        if looks_like_profile_name_only(text, user=user):
            continue
        if len(text) < 2:
            continue
        dedupe_key = (user.lower(), text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append({"user": user, "text": text, "timestamp": timestamp})
    return cleaned


async def jitter_pause(page, low_ms=350, high_ms=850):
    await page.wait_for_timeout(random.randint(low_ms, high_ms))


async def get_facebook_page_and_context(browser, target_url):
    contexts = browser.contexts
    if contexts:
        context = contexts[0]
    else:
        context = await browser.new_context()
    for page in context.pages:
        if "facebook.com" in page.url:
            return context, page
    page = await context.new_page()
    await page.goto(target_url, wait_until="domcontentloaded")
    return context, page


async def close_facebook_popup_x(page):
    close_selectors = [
        '[role="dialog"] [aria-label="Fermer"]',
        '[role="dialog"] [aria-label="Close"]',
        '[role="dialog"] [aria-label="Schließen"]',
        '[aria-modal="true"] [aria-label="Fermer"]',
        '[aria-modal="true"] [aria-label="Close"]',
        '[aria-modal="true"] [aria-label="Schließen"]',
        '[role="dialog"] [data-testid="cookie-policy-dialog-close-button"]',
    ]
    for selector in close_selectors:
        locator = page.locator(selector).first
        try:
            if await locator.count():
                await locator.scroll_into_view_if_needed(timeout=1000)
                box = await locator.bounding_box()
                if box:
                    await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=8)
                    await jitter_pause(page, 120, 260)
                    await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                    await jitter_pause(page, 450, 900)
                    logger.info(f"Closed Facebook popup via visible X button using selector: {selector}")
                    return True
        except Exception:
            continue

    try:
        dialog = page.locator('[role="dialog"], [aria-modal="true"]').first
        if await dialog.count():
            buttons = dialog.locator('div[role="button"], button')
            count = await buttons.count()
            for idx in range(min(count, 12)):
                candidate = buttons.nth(idx)
                text = normalize_scraped_text(await candidate.inner_text())
                if text in {"X", "x", "×", "✕"}:
                    box = await candidate.bounding_box()
                    if box:
                        await page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=8)
                        await jitter_pause(page, 120, 260)
                        await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        await jitter_pause(page, 450, 900)
                        logger.info("Closed Facebook popup via glyph X button.")
                        return True
    except Exception:
        pass
    return False


async def recover_mobile_post_page(page, original_url):
    page_text = normalize_scraped_text(await page.locator("body").inner_text()).lower()
    page_title = (await page.title()).lower()
    unavailable_markers = [
        "ce contenu n'est pas disponible hors connexion",
        "this content isn't available unless you log in",
        "page indisponible",
        "page unavailable",
    ]
    if any(marker in page_text for marker in unavailable_markers) or any(marker in page_title for marker in unavailable_markers):
        logger.info(f"Facebook mobile tab drifted to unavailable page ({page.url}); reloading original post.")
        await page.goto(original_url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
        return True
    return False


async def try_select_all_comments(page):
    sort_buttons = [
        page.get_by_role("button", name=re.compile(r"Most relevant|Plus pertinents|Relevanteste", re.I)),
        page.get_by_text(re.compile(r"Most relevant|Plus pertinents|Relevanteste", re.I)),
    ]
    option_patterns = re.compile(r"All comments|Tous les commentaires|Alle Kommentare", re.I)

    for button in sort_buttons:
        try:
            if await button.count():
                await button.first.click(timeout=1500)
                await page.wait_for_timeout(700)
                option = page.get_by_text(option_patterns).first
                if await option.count():
                    await option.click(timeout=1500)
                    await page.wait_for_timeout(1200)
                    logger.info("Facebook comment sort switched to All comments.")
                    return True
        except Exception:
            continue
    return False


async def wait_for_facebook_loading_to_finish(page, max_rounds=5):
    """Wait for Facebook's dynamic content to finish rendering specialized for comments."""
    for i in range(max_rounds):
        # We check for specific visible loading elements or text within the main content area
        # Checking the entire body text is too prone to false positives from static footers/sidebars
        loading_visible = False
        try:
            # Check for standard Facebook loading markers if they are actually visible
            specific_markers = [
                "loading comments", "chargement des commentaires", "kommentare werden geladen",
                "loading more comments", "chargement...", "veuillez patienter", "patientez"
            ]
            for marker in specific_markers:
                # Use a locator that only looks for visible elements containing this text
                if await page.get_by_text(re.compile(marker, re.I)).first.is_visible():
                    loading_visible = True
                    break
            
            # Also check for progress bars or explicit loading indicators
            if not loading_visible:
                loading_visible = await page.locator('[role="progressbar"], svg[aria-label*="Loading"], div[aria-busy="true"]').first.is_visible()
        except Exception:
            loading_visible = False
            
        if not loading_visible:
            if i > 0:
                logger.info("Facebook loading state cleared.")
            break
            
        logger.info(f"Detected Facebook loading state (round {i+1}); waiting for comments to render...")
        await page.wait_for_timeout(1000)


async def extract_structured_comments_from_page(page):
    records = await page.evaluate(
        """
        () => {
            const normalize = (value) => (value || '').replace(/\\u00a0/g, ' ').replace(/\\u200b/g, ' ').replace(/\\s+/g, ' ').trim();
            const splitLines = (value) => normalize(value).split(/\\n+/).map(normalize).filter(Boolean);
            const uiPattern = /^(view \\d+ repl|view more|view previous|afficher \\d+ r|afficher plus|charger plus|most relevant|all comments|newest|comments?|likes?|replies|j['’]?aime|répondre|reply)$/i;
            const endPattern = /(recent posts|publications récentes|related pages)/i;
            const timePattern = /(^|\\b)(\\d+\\s*(s|min|h|d|j|w|sem|mo|an|ans|yr|yrs)|hier|yesterday|today|aujourd'hui|gestern|heute)(\\b|$)/i;
            const visible = (el) => {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style && style.visibility !== 'hidden' && style.display !== 'none' && rect.width > 0 && rect.height > 0;
            };
            const records = [];
            const seen = new Set();
            const anchors = Array.from(document.querySelectorAll('a[href]')).filter(visible);

            const getCandidateFromRoot = (root, userHint) => {
                if (!root || !visible(root)) return null;
                const lines = splitLines(root.innerText).slice(0, 10);
                if (lines.length < 2) return null;
                if (endPattern.test(lines.join(' '))) return null;
                const user = normalize(userHint || lines[0]);
                if (!user || user.length > 60) return null;
                let timestamp = 'Unknown';
                const body = [];
                for (const line of lines) {
                    if (!line || line === user) continue;
                    if (uiPattern.test(line)) continue;
                    if (timePattern.test(line) && timestamp === 'Unknown') {
                        timestamp = line;
                        continue;
                    }
                    if (line.length >= 2) {
                        body.push(line);
                    }
                }
                const text = normalize(body.join(' '));
                if (!text || text === user || text.length > 450) return null;
                return { user, text, timestamp };
            };

            for (const anchor of anchors) {
                const user = normalize(anchor.innerText);
                if (!user || user.length < 2 || user.length > 60) continue;
                if (/^(facebook|connexion|se connecter|log in|login|ramy food)$/i.test(user)) continue;
                let root = anchor;
                for (let depth = 0; depth < 6 && root; depth += 1) {
                    root = root.parentElement;
                    const candidate = getCandidateFromRoot(root, user);
                    if (!candidate) continue;
                    const key = `${candidate.user}||${candidate.text}`;
                    if (!seen.has(key)) {
                        seen.add(key);
                        records.push(candidate);
                    }
                    break;
                }
            }

            return records;
        }
        """
    )
    return normalize_comment_records(records)


def extract_comments_from_body_text(body_text):
    lines = [normalize_scraped_text(line) for line in body_text.splitlines()]
    lines = [line for line in lines if line]
    records = []
    i = 0

    while i < len(lines):
        line = lines[i]
        lowered = line.lower()

        if any(marker in lowered for marker in FACEBOOK_END_MARKERS):
            break
        if lowered in {"super fan", "top fan"} or line.startswith("󱘫"):
            i += 1
            continue
        if FACEBOOK_TIMESTAMP_RE.search(line) or is_ui_residue_comment(line):
            i += 1
            continue
        if re.fullmatch(r"[\d.,]+\s*[kKmM]?", line):
            i += 1
            continue
        if len(line) > 80:
            i += 1
            continue
        if not re.search(r"[A-Za-zÀ-ÿ\u0600-\u06FF]", line):
            i += 1
            continue
        if "autres personnes" in lowered or "other people" in lowered or lowered.startswith("afficher les réponses"):
            i += 1
            continue

        timestamp_idx = None
        for j in range(i + 1, min(i + 7, len(lines))):
            probe = lines[j]
            if any(marker in probe.lower() for marker in FACEBOOK_END_MARKERS):
                break
            if FACEBOOK_TIMESTAMP_RE.search(probe):
                timestamp_idx = j
                break

        if timestamp_idx is None:
            i += 1
            continue

        user = line
        text_lines = []
        for k in range(i + 1, timestamp_idx):
            current = lines[k]
            current_lower = current.lower()
            if current_lower in {"super fan", "top fan"} or current.startswith("󱘫"):
                continue
            if is_ui_residue_comment(current):
                continue
            if re.fullmatch(r"[\d.,]+\s*[kKmM]?", current):
                continue
            if "autres personnes" in current_lower or current_lower.startswith("afficher les réponses"):
                continue
            if not re.search(r"[A-Za-zÀ-ÿ\u0600-\u06FF]", current):
                continue
            text_lines.append(current)

        text = clean_comment_text(" ".join(text_lines))
        if text and not looks_like_profile_name_only(text, user=user):
            records.append({"user": user, "text": text, "timestamp": lines[timestamp_idx]})

        i = timestamp_idx + 1
        while i < len(lines):
            trailing = lines[i]
            trailing_lower = trailing.lower()
            if trailing_lower in {"super fan", "top fan"} or trailing.startswith("󱘫"):
                i += 1
                continue
            if re.fullmatch(r"[\d.,]+\s*[kKmM]?", trailing):
                i += 1
                continue
            if is_ui_residue_comment(trailing):
                i += 1
                continue
            break

    return normalize_comment_records(records)


async def scrape_facebook_comments_mobile(url):
    target_url = to_www_facebook_url(url)
    all_unique_comments = []
    seen_global = set()

    for profile in MOBILE_BROWSER_PROFILES:
        runtime = launch_real_chrome_for_cdp(target_url, profile, mobile_mode=True)
        try:
            async with async_playwright() as p:
                browser = await p.chromium.connect_over_cdp(f"http://127.0.0.1:{runtime['port']}")
                context, page = await get_facebook_page_and_context(browser, target_url)
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_timeout(1600)
                try:
                    await context.set_extra_http_headers({"Accept-Language": profile.get("locale", "fr-FR")})
                except Exception:
                    pass

                await close_facebook_popup_x(page)
                await try_select_all_comments(page)

                all_unique_comments = []
                seen_global = set()
                stable_rounds = 0
                max_stable_threshold = 40
                for _ in range(150):
                    await recover_mobile_post_page(page, target_url)
                    await close_facebook_popup_x(page)
                    await wait_for_facebook_loading_to_finish(page)

                    # Periodically try to find and click "view more" style buttons
                    try:
                        view_more_labels = [
                            "view more comments", "plus de commentaires", "afficher plus de commentaires", 
                            "view previous comments", "voir les commentaires précédents",
                            "réponse", "réponses", "reply", "replies", "afficher les", "view all", "plus de",
                            "voir plus", "voir tout"
                        ]
                        for label in view_more_labels:
                            btns = page.get_by_text(re.compile(label, re.I))
                            count = await btns.count()
                            if count > 0:
                                for btn_idx in range(min(count, 3)):
                                    btn = btns.nth(btn_idx)
                                    if await btn.is_visible():
                                        await btn.click(timeout=1000)
                                        await page.wait_for_timeout(800)
                                        logger.info(f"Clicked 'view more' button hit: {label}")
                                        break
                    except Exception:
                        pass

                    body_text = await page.locator("body").inner_text()
                    current_batch = extract_comments_from_body_text(body_text)
                    if len(current_batch) < 6:
                        dom_comments = await extract_structured_comments_from_page(page)
                        if len(dom_comments) > len(current_batch):
                            current_batch = dom_comments
                    
                    # Accumulate unique comments globally
                    new_found = 0
                    for c in current_batch:
                        key = (c["user"].lower(), c["text"].lower())
                        if key not in seen_global:
                            all_unique_comments.append(c)
                            seen_global.add(key)
                            new_found += 1
                    
                    if new_found > 0:
                        stable_rounds = 0
                        logger.info(f"Found {new_found} new unique comments (Total: {len(all_unique_comments)})")
                    else:
                        stable_rounds += 1

                    lowered_body = normalize_scraped_text(body_text).lower()
                    if any(marker in lowered_body for marker in FACEBOOK_END_MARKERS):
                        logger.info("Detected end-of-thread markers in Facebook mobile view; stopping scroll loop.")
                        break

                    viewport_height = profile["viewport"]["height"]
                    # Perform multiple small scrolls to trigger lazy loading better
                    for _ in range(2):
                        await page.mouse.wheel(0, int(viewport_height * 0.8))
                        await page.wait_for_timeout(800)
                    
                    # Force a JS scroll as well
                    await page.evaluate("window.scrollBy(0, 1000)")
                    await page.wait_for_timeout(1500)
                    await wait_for_facebook_loading_to_finish(page)

                    if any(marker in lowered_body for marker in FACEBOOK_GATE_MARKERS):
                        close_attempted = await close_facebook_popup_x(page)
                        logger.info(f"detected login gate; close attempted={close_attempted} via label-button")
                        await recover_mobile_post_page(page, target_url)

                    # Only stop if we've been stable for a long time AND we have at least some comments
                    # or if we've been stable for a VERY long time and have nothing.
                    if stable_rounds >= max_stable_threshold:
                        if len(all_unique_comments) > 25 or stable_rounds >= 60:
                            logger.info(f"Stabilized for {stable_rounds} rounds; stopping loop.")
                            break
                        else:
                            logger.info(f"Stable for {stable_rounds} rounds but only have {len(all_unique_comments)} comments; continuing search...")

                await browser.close()
        finally:
            cleanup_chrome_runtime(runtime)

    return all_unique_comments


async def scrape_facebook_comments_better_way(url):
    """
    A more robust, desktop-based scraper that handles login modals and 'All Comments' filtering.
    """
    logger.info(f"Using 'Better Way' scraper for: {url}")
    target_url = to_www_facebook_url(url)
    all_unique_comments = []
    seen_global = set()

    # Reuse the runtime detection and virtual display logic
    runtime = {"display": maybe_start_virtual_display(1280, 800)}
    
    try:
        async with async_playwright() as p:
            # Launch standard Chromium with stealth-like arguments
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled", "--disable-infobars"
                ]
            )
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                locale="fr-FR"
            )
            page = await context.new_page()
            
            logger.info(f"Navigating to Desktop URL: {target_url}")
            await page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(3000)
            
            # 1. Bypass Login Modal / Overlay
            await page.evaluate("""
                () => {
                    const selectors = ['div[role="dialog"]', 'div#login_popup', 'div#header_block', 'div.x1n2onr6.x1vjfegm'];
                    selectors.forEach(s => {
                        const elements = document.querySelectorAll(s);
                        elements.forEach(el => {
                            if (el && (el.innerText.toLowerCase().includes('se connecter') || el.innerText.toLowerCase().includes('log in'))) {
                                el.remove();
                            }
                        });
                    });
                    document.body.style.overflow = 'auto'; // Re-enable scrolling
                }
            """)
            
            # Try to click explicit 'X' if visible
            for x_selector in ['div[aria-label="Fermer"]', 'div[aria-label="Close"]', 'div[role="button"]:has-text("X")']:
                try:
                    btn = page.locator(x_selector).first
                    if await btn.count() > 0 and await btn.is_visible():
                        await btn.click(timeout=2000)
                        logger.info(f"Closed login modal via {x_selector}")
                except: pass

            # 2. Select "All Comments" Filter
            filter_labels = ["Plus pertinents", "Most relevant", "Commentaires"]
            for label in filter_labels:
                try:
                    btn = page.get_by_text(label).first
                    if await btn.count() > 0 and await btn.is_visible():
                        await btn.click()
                        await page.wait_for_timeout(1500)
                        # Select "All"
                        all_options = [
                            "Toutes les réponses", "All comments", "Tous les commentaires",
                            "Les plus récents", "Newest"
                        ]
                        for al in all_options:
                            all_btn = page.get_by_text(al).first
                            if await all_btn.count() > 0:
                                await all_btn.click()
                                logger.info(f"Selected '{al}' filter.")
                                await page.wait_for_timeout(2500)
                                break
                        break
                except: pass

            # 3. Recursive Expansion & Extraction
            stable_rounds = 0
            for scroll_idx in range(120):
                # Click "View more" buttons recursively
                more_labels = [
                    "Plus de commentaires", "View more comments", "réponses", "replies", 
                    "voir plus", "view all", "afficher les"
                ]
                for ml in more_labels:
                    try:
                        btns = page.get_by_text(re.compile(ml, re.I))
                        count = await btns.count()
                        if count > 0:
                            for i in range(min(count, 8)): # Click up to 8 buttons per round
                                b = btns.nth(i)
                                if await b.is_visible():
                                    await b.click(timeout=1200)
                                    await page.wait_for_timeout(800)
                    except: pass
                
                # Extract using internal DOM heuristic
                batch = await extract_structured_comments_from_page(page)
                
                # Also fallback to body text if DOM fails
                body_text = await page.locator("body").inner_text()
                body_batch = extract_comments_from_body_text(body_text)
                
                new_found = 0
                for c in (batch + body_batch):
                    key = (c["user"].lower(), c["text"].lower())
                    if key not in seen_global:
                        all_unique_comments.append(c)
                        seen_global.add(key)
                        new_found += 1
                
                if new_found > 0:
                    stable_rounds = 0
                    logger.info(f"Better Way: Found {new_found} new (Total: {len(all_unique_comments)})")
                else:
                    stable_rounds += 1
                
                # Scroll the page or the active dialog if one exists
                await page.evaluate("window.scrollBy(0, 1000)")
                await page.wait_for_timeout(2000)
                
                if stable_rounds >= 35 and len(all_unique_comments) > 15:
                    break
            
            await browser.close()
    except Exception as e:
        logger.error(f"Better Way Error: {e}")
    finally:
        cleanup_chrome_runtime(runtime)
        
    return all_unique_comments


async def scrape_and_extract_comments(url):
    logger.info(f"Starting scrape_and_extract_comments for URL: {url}")

    if "facebook.com" in url or "fb.com" in url:
        try:
            # 1. Primary: Better Way (Desktop Stealth)
            comments = await scrape_facebook_comments_better_way(url)
            
            # 2. Fallback: Mobile
            if not comments or len(comments) < 5:
                logger.info("Better Way yielded too few; attempting Mobile fallback...")
                mobile_comments = await scrape_facebook_comments_mobile(url)
                if mobile_comments and len(mobile_comments) > len(comments or []):
                    comments = mobile_comments
            
            if comments:
                logger.info(f"Facebook extraction complete: {len(comments)} comments total.")
                return json.dumps(comments)
            return None
        except Exception as e:
            logger.error(f"Scrape execution error: {e}")
            return None
    return None


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
            user = item.get("user", "")
        else:
            text = str(item)
            timestamp = "Unknown"
            user = ""
            
        text = clean_comment_text(text)
        if not text or not text.strip():
            continue
        if looks_like_profile_name_only(text, user=user):
            logger.info(f"Skipping name-only candidate at position {idx+1}: {text}")
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

def format_ai_brief_error(error):
    message = str(error)
    lowered = message.lower()
    if "memory layout cannot be allocated" in lowered or "out of memory" in lowered:
        return "⚠️ Could not generate AI brief: Ollama is running, but the local model ran out of memory. Close other heavy apps, restart Ollama, or use a smaller Ollama model."
    if "apiconnectionerror" in lowered or "connection refused" in lowered or "failed to connect" in lowered:
        return "⚠️ Could not generate AI brief: Ollama is not reachable on http://localhost:11434."
    return f"⚠️ Could not generate AI brief: {message}"

def generate_executive_brief(df):
    try:
        top_issues_all = df[df["High Priority"] == "🚨 YES"]["Comment"].tolist()
        general_all = df["Comment"].tolist()
        attempts = [
            {"top_n": 8, "general_n": 12, "num_ctx": 2048},
            {"top_n": 5, "general_n": 8, "num_ctx": 1024},
            {"top_n": 3, "general_n": 5, "num_ctx": 768},
        ]
        last_error = None

        for attempt in attempts:
            top_issues = top_issues_all[:attempt["top_n"]]
            general = general_all[:attempt["general_n"]]
            prompt = (
                "You are an expert Social Media Sentiment Analyst. I collected comments from Algerian users on a recent post. "
                "Write a strict 3-sentence executive brief. Mention the main source of frustration or anger if present, "
                "the overall sentiment, and one short business takeaway. Keep it concise and professional.\n\n"
                f"High Priority Comments: {top_issues}\n"
                f"General Comments: {general}"
            )
            try:
                response = litellm.completion(
                    model="ollama/gemma3:4b",
                    messages=[{"role": "user", "content": prompt}],
                    api_base="http://localhost:11434",
                    num_ctx=attempt["num_ctx"],
                )
                return response.choices[0].message.content.strip()
            except Exception as error:
                last_error = error
                logger.warning(
                    f"AI brief attempt failed (num_ctx={attempt['num_ctx']}, top_n={attempt['top_n']}, general_n={attempt['general_n']}): {error}"
                )

        raise last_error
    except Exception as e:
        logger.error(f"Error generating AI Brief: {e}")
        return format_ai_brief_error(e)


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
        st.plotly_chart(fig_donut, width='stretch')

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
        st.plotly_chart(fig_bar, width='stretch')

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
                    st.write("Launching a full browser-driven scraper and extracting comments from the live page DOM...")
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
                    data_payload = [
                        {
                            "text": item.get("text", str(item)),
                            "timestamp": item.get("timestamp", "Unknown"),
                            "user": item.get("user", ""),
                        }
                        for item in comments_list
                    ]
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
                        st.dataframe(df_csv[[col_name]].head(8), width='stretch')
                        
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
