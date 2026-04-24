import streamlit as st
import os
import re
import requests
import sqlite3
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield — Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    color: #e8eaf6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 40px 36px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(102,126,234,0.35);
    text-align: center;
}
.hero-banner h1 { font-size: 2.8rem; font-weight: 800; color: white; margin: 0; letter-spacing: -1px; }
.hero-banner p  { font-size: 1.05rem; color: rgba(255,255,255,0.85); margin-top: 8px; }

/* Result cards */
.verdict-card {
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.verdict-real    { background: linear-gradient(135deg, #11998e, #38ef7d); }
.verdict-fake    { background: linear-gradient(135deg, #c0392b, #e74c3c); }
.verdict-suspicious { background: linear-gradient(135deg, #f39c12, #f1c40f); }

.verdict-card h2 { font-size: 2rem; font-weight: 800; color: white; margin: 0; }
.verdict-card p  { color: rgba(255,255,255,0.9); font-size: 1rem; margin-top: 6px; }

/* Metric tiles */
.metric-tile {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
}
.metric-tile .label { font-size: 0.78rem; color: #a0a8c0; text-transform: uppercase; letter-spacing: 1px; }
.metric-tile .value { font-size: 1.9rem; font-weight: 700; color: #e8eaf6; margin-top: 4px; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(102,126,234,0.6) !important;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e8eaf6 !important;
    font-size: 0.95rem !important;
}

/* Section headers */
.section-header {
    font-size: 1.2rem;
    font-weight: 700;
    color: #a78bfa;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid rgba(167,139,250,0.3);
}

/* Info box */
.info-box {
    background: rgba(102,126,234,0.12);
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c7d2fe;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* Warning flags */
.flag-item {
    display: inline-block;
    background: rgba(239,68,68,0.15);
    border: 1px solid rgba(239,68,68,0.4);
    color: #fca5a5;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Feature chips */
.feature-chip {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.35);
    color: #c4b5fd;
    border-radius: 20px;
    padding: 4px 14px;
    margin: 4px;
    font-size: 0.82rem;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: #a0a8c0 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Helpers ────────────────────────────────────────────────────────────────────
DB_PATH = "fake_news_system.db"
MODEL_PATH = "models/fake_news_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        news_text TEXT, predicted_label TEXT,
        user_label TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    c.execute('''CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name TEXT UNIQUE, trust_score REAL DEFAULT 50.0, verified INTEGER DEFAULT 0)''')
    for src, score, v in [('BBC',95,1),('Reuters',98,1),('The Hindu',90,1),
                           ('AP News',96,1),('NDTV',80,1),('Daily Mail',38,0),
                           ('WhatsApp Forward',10,0),('Unknown',30,0)]:
        c.execute("INSERT OR IGNORE INTO sources(source_name,trust_score,verified) VALUES(?,?,?)",(src,score,v))
    conn.commit(); conn.close()

def save_feedback(text, predicted, user_label):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO feedback(news_text,predicted_label,user_label) VALUES(?,?,?)",(text,predicted,user_label))
    conn.commit(); conn.close()

def get_source_score(name):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT trust_score FROM sources WHERE source_name=?",(name,)).fetchone()
    conn.close()
    return r[0] if r else 50.0

def get_feedback_stats():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT predicted_label, user_label FROM feedback").fetchall()
    conn.close()
    return rows

# ─── NLP & Analysis ─────────────────────────────────────────────────────────────
CLICKBAIT_PATTERNS = [
    r"you won'?t believe", r"shocking", r"breaking news", r"miracle",
    r"doctors hate", r"one weird trick", r"this changes everything",
    r"must see", r"going viral", r"secret revealed", r"100%", r"cure",
    r"exclusive", r"exposed", r"conspiracy",
]
EMOTIONAL_WORDS = [
    "outrage","panic","fear","catastrophe","disaster","scandal","hoax",
    "attack","explosion","threat","warn","crisis","danger","urgent","alert",
]
ALL_CAPS_RE = re.compile(r'\b[A-Z]{4,}\b')
EXCLAMATION_RE = re.compile(r'!')
URL_RE = re.compile(r'https?://\S+|www\.\S+')

def detect_clickbait(text):
    t = text.lower()
    found = [p for p in CLICKBAIT_PATTERNS if re.search(p, t)]
    return found

def detect_emotional_manipulation(text):
    t = text.lower()
    found = [w for w in EMOTIONAL_WORDS if w in t]
    caps_count = len(ALL_CAPS_RE.findall(text))
    excl_count = len(EXCLAMATION_RE.findall(text))
    score = min(100, len(found) * 12 + caps_count * 8 + excl_count * 5)
    return found, caps_count, excl_count, score

def simple_clean(text):
    t = text.lower()
    t = URL_RE.sub('', t)
    t = re.sub(r'[^a-zA-Z\u0B80-\u0BFF\s]', '', t)
    return t.strip()

def detect_language(text):
    tamil_chars = len(re.findall(r'[\u0B80-\u0BFF]', text))
    return "Tamil" if tamil_chars > len(text) * 0.1 else "English"

def classify_news(text, model, source_name="Unknown"):
    if not text.strip():
        return None

    language = detect_language(text)
    cleaned = simple_clean(text)

    # ML prediction
    ml_label, ml_prob = "Unknown", [0.33, 0.33, 0.34]
    if model:
        try:
            prob = model.predict_proba([cleaned])[0]
            classes = model.classes_
            idx = np.argmax(prob)
            ml_label = classes[idx]
            ml_prob = prob
        except Exception:
            pass

    # Heuristic signals
    cb_flags = detect_clickbait(text)
    emo_words, caps, excl, emo_score = detect_emotional_manipulation(text)
    source_score = get_source_score(source_name)

    # Credibility score (0-100)
    base = {"Real": 78, "Fake": 22, "Suspicious": 48, "Unknown": 50}.get(ml_label, 50)
    credibility = base
    credibility -= len(cb_flags) * 6
    credibility -= emo_score * 0.2
    credibility += (source_score - 50) * 0.3
    credibility = max(0, min(100, credibility))

    # Final verdict
    if credibility >= 65:
        verdict = "REAL"
        verdict_class = "verdict-real"
        verdict_emoji = "✅"
    elif credibility <= 35:
        verdict = "FAKE"
        verdict_class = "verdict-fake"
        verdict_emoji = "❌"
    else:
        verdict = "SUSPICIOUS"
        verdict_class = "verdict-suspicious"
        verdict_emoji = "⚠️"

    # Top keywords (simple frequency)
    words = cleaned.split()
    freq = {}
    stop = {"the","a","an","is","it","in","of","to","and","for","this","that",
            "was","are","with","be","at","by","from","or","on","as","have","has"}
    for w in words:
        if len(w) > 3 and w not in stop:
            freq[w] = freq.get(w, 0) + 1
    top_words = sorted(freq, key=freq.get, reverse=True)[:8]

    return {
        "verdict": verdict,
        "verdict_class": verdict_class,
        "verdict_emoji": verdict_emoji,
        "credibility": round(credibility, 1),
        "ml_label": ml_label,
        "ml_prob": ml_prob,
        "ml_classes": model.classes_ if model else ["Fake","Real","Suspicious"],
        "language": language,
        "clickbait_flags": cb_flags,
        "emotional_words": emo_words,
        "caps_count": caps,
        "excl_count": excl,
        "emotional_score": emo_score,
        "source_score": source_score,
        "top_words": top_words,
        "word_count": len(words),
    }

def fetch_related_news(query):
    """Try NewsAPI; return empty if no key or error."""
    api_key = os.environ.get("NEWS_API_KEY", "")
    if not api_key:
        return []
    try:
        r = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": query[:80], "pageSize": 4, "apiKey": api_key},
            timeout=5,
        )
        if r.status_code == 200:
            arts = r.json().get("articles", [])
            return [{"title": a["title"], "source": a["source"]["name"], "url": a["url"]} for a in arts]
    except Exception:
        pass
    return []

# ─── UI Layout ──────────────────────────────────────────────────────────────────
init_db()
model = load_model()

# Hero
st.markdown("""
<div class="hero-banner">
  <h1>🛡️ FakeShield</h1>
  <p>Intelligent Fake News Detection · Explainable AI · English & Tamil</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    source_name = st.selectbox(
        "📰 News Source",
        ["Unknown","BBC","Reuters","AP News","The Hindu","NDTV","Daily Mail","WhatsApp Forward"],
        index=0,
    )
    language_hint = st.radio("🌐 Language", ["Auto-detect","English","Tamil"], index=0)
    st.markdown("---")
    st.markdown("### 📊 Model Status")
    if model:
        st.success("✅ ML Model loaded")
    else:
        st.warning("⚠️ No trained model found\nRun `python model_trainer.py` first")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    <div class="info-box">
    FakeShield combines:<br>
    🤖 ML Classification (TF-IDF + RF)<br>
    🧠 Clickbait Detection<br>
    💢 Emotional Analysis<br>
    📡 Real-time Fact Check<br>
    🗣️ Tamil + English support
    </div>
    """, unsafe_allow_html=True)

# Main tabs
tab_detect, tab_history, tab_about = st.tabs(["🔍 Detect", "📜 History", "📘 How It Works"])

# ── Tab 1: Detect ──────────────────────────────────────────────────────────────
with tab_detect:
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown('<p class="section-header">📝 Enter News Article</p>', unsafe_allow_html=True)
        news_text = st.text_area(
            label="",
            placeholder="Paste a news headline or full article here...",
            height=200,
            key="news_input",
        )

        # Sample buttons
        st.markdown("**Try a sample:**")
        sc1, sc2, sc3 = st.columns(3)
        sample_en_real = "Scientists at NASA have confirmed the discovery of water ice deposits near the lunar south pole, boosting hopes for future Moon missions."
        sample_en_fake = "BREAKING: Drinking lemon water CURES cancer completely overnight! Doctors are SHOCKED by this secret cure they've been hiding from you!!!"
        sample_ta = "அரசு புதிய திட்டத்தை அறிவிக்கிறது. விவசாயிகளுக்கு நேரடி நிதி உதவி வழங்கப்படும்."
        if sc1.button("✅ Real (EN)", use_container_width=True):
            st.session_state["prefill"] = sample_en_real
            news_text = sample_en_real
        if sc2.button("❌ Fake (EN)", use_container_width=True):
            st.session_state["prefill"] = sample_en_fake
            news_text = sample_en_fake
        if sc3.button("🇮🇳 Tamil", use_container_width=True):
            st.session_state["prefill"] = sample_ta
            news_text = sample_ta

        if "prefill" in st.session_state and not news_text:
            news_text = st.session_state["prefill"]

        analyze_btn = st.button("🚀 Analyze Article", use_container_width=True)

    with col_out:
        if analyze_btn and news_text.strip():
            result = classify_news(news_text, model, source_name)
            if result:
                # Verdict card
                st.markdown(f"""
                <div class="verdict-card {result['verdict_class']}">
                  <h2>{result['verdict_emoji']} {result['verdict']}</h2>
                  <p>Language: {result['language']} · Words: {result['word_count']}</p>
                </div>
                """, unsafe_allow_html=True)

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.markdown(f"""<div class="metric-tile">
                  <div class="label">Credibility</div>
                  <div class="value">{result['credibility']}%</div>
                </div>""", unsafe_allow_html=True)
                m2.markdown(f"""<div class="metric-tile">
                  <div class="label">Emotional Score</div>
                  <div class="value">{result['emotional_score']}</div>
                </div>""", unsafe_allow_html=True)
                m3.markdown(f"""<div class="metric-tile">
                  <div class="label">Source Trust</div>
                  <div class="value">{result['source_score']:.0f}%</div>
                </div>""", unsafe_allow_html=True)

                st.markdown("---")

                # Gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=result['credibility'],
                    title={'text': "Credibility Score", 'font': {'color': '#e8eaf6', 'size': 14}},
                    gauge={
                        'axis': {'range': [0, 100], 'tickcolor': '#a0a8c0'},
                        'bar': {'color': '#667eea'},
                        'steps': [
                            {'range': [0, 35], 'color': 'rgba(231,76,60,0.3)'},
                            {'range': [35, 65], 'color': 'rgba(241,196,15,0.3)'},
                            {'range': [65, 100], 'color': 'rgba(56,239,125,0.3)'},
                        ],
                        'threshold': {'line': {'color': '#f093fb', 'width': 3}, 'value': result['credibility']},
                    },
                    number={'suffix': '%', 'font': {'color': '#e8eaf6', 'size': 28}},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e8eaf6',
                    height=250,
                    margin=dict(t=40, b=10, l=20, r=20),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

                # Clickbait flags
                if result['clickbait_flags']:
                    st.markdown('<p class="section-header">🚩 Clickbait Indicators</p>', unsafe_allow_html=True)
                    flags_html = "".join(f'<span class="flag-item">⚠ {f}</span>' for f in result['clickbait_flags'])
                    st.markdown(flags_html, unsafe_allow_html=True)

                # Emotional words
                if result['emotional_words']:
                    st.markdown('<p class="section-header">💢 Emotional Manipulation</p>', unsafe_allow_html=True)
                    words_html = "".join(f'<span class="flag-item">{w}</span>' for w in result['emotional_words'])
                    st.markdown(words_html, unsafe_allow_html=True)
                    if result['caps_count']:
                        st.caption(f"🔠 {result['caps_count']} ALL-CAPS words · ❗ {result['excl_count']} exclamation marks")

                # Key features
                if result['top_words']:
                    st.markdown('<p class="section-header">🔑 Key Terms</p>', unsafe_allow_html=True)
                    chips = "".join(f'<span class="feature-chip">{w}</span>' for w in result['top_words'])
                    st.markdown(chips, unsafe_allow_html=True)

                # ML probability bar chart
                if model and result['ml_label'] != "Unknown":
                    st.markdown('<p class="section-header">📊 ML Model Confidence</p>', unsafe_allow_html=True)
                    classes = list(result['ml_classes'])
                    probs = [round(p * 100, 1) for p in result['ml_prob']]
                    colors = []
                    for c in classes:
                        if c == "Real": colors.append('#38ef7d')
                        elif c == "Fake": colors.append('#e74c3c')
                        else: colors.append('#f1c40f')
                    fig_bar = go.Figure(go.Bar(
                        x=classes, y=probs,
                        marker_color=colors,
                        text=[f"{p}%" for p in probs],
                        textposition='auto',
                    ))
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#e8eaf6',
                        height=220,
                        margin=dict(t=10, b=10, l=20, r=20),
                        yaxis=dict(range=[0, 100], gridcolor='rgba(255,255,255,0.08)'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.08)'),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Real-time fact check
                st.markdown('<p class="section-header">📡 Related News (Fact Check)</p>', unsafe_allow_html=True)
                keywords = " ".join(result['top_words'][:3])
                related = fetch_related_news(keywords)
                if related:
                    for art in related:
                        st.markdown(f"- **{art['source']}**: [{art['title']}]({art['url']})")
                else:
                    st.markdown("""<div class="info-box">
                    💡 Set the <code>NEWS_API_KEY</code> environment variable to enable real-time fact checking via NewsAPI.org
                    </div>""", unsafe_allow_html=True)

                # Feedback
                st.markdown("---")
                st.markdown('<p class="section-header">🗣️ Was this prediction correct?</p>', unsafe_allow_html=True)
                fb_col1, fb_col2, fb_col3 = st.columns(3)
                if fb_col1.button("✅ Yes, Correct"):
                    save_feedback(news_text[:500], result['verdict'], result['verdict'])
                    st.success("Thanks for your feedback!")
                if fb_col2.button("❌ No, It's Real"):
                    save_feedback(news_text[:500], result['verdict'], "REAL")
                    st.info("Noted! Feedback saved.")
                if fb_col3.button("⚠️ No, It's Fake"):
                    save_feedback(news_text[:500], result['verdict'], "FAKE")
                    st.info("Noted! Feedback saved.")

        elif analyze_btn:
            st.warning("Please enter some news text to analyze.")
        else:
            st.markdown("""<div class="info-box" style="margin-top:80px;text-align:center;">
            👈 Enter a news article and click <strong>Analyze Article</strong>
            </div>""", unsafe_allow_html=True)

# ── Tab 2: History ─────────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<p class="section-header">📜 Feedback History & Model Learning</p>', unsafe_allow_html=True)
    rows = get_feedback_stats()
    if rows:
        import pandas as pd
        df = pd.DataFrame(rows, columns=["Predicted", "User Label"])
        correct = sum(1 for r in rows if r[0] == r[1])
        total = len(rows)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Feedback", total)
        c2.metric("Correct Predictions", correct)
        c3.metric("Accuracy", f"{correct/total*100:.1f}%" if total else "N/A")

        counts = df["User Label"].value_counts().reset_index()
        counts.columns = ["Label", "Count"]
        fig = px.pie(counts, names="Label", values="Count",
                     color_discrete_sequence=["#38ef7d","#e74c3c","#f1c40f","#667eea"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8eaf6', height=300)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.tail(20), use_container_width=True)
    else:
        st.markdown('<div class="info-box">No feedback yet. Analyze some articles and provide feedback!</div>', unsafe_allow_html=True)

# ── Tab 3: How It Works ────────────────────────────────────────────────────────
with tab_about:
    st.markdown('<p class="section-header">📘 System Architecture</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>FakeShield</strong> uses a multi-layered detection approach:<br><br>
    <strong>1. 🤖 ML Classification</strong><br>
    TF-IDF vectorization + Random Forest Classifier trained on English & Tamil articles.
    Outputs probability scores for Real / Fake / Suspicious.<br><br>
    <strong>2. 🚩 Clickbait Detection</strong><br>
    Pattern-matching against 15+ known clickbait phrases and sensationalist language indicators.<br><br>
    <strong>3. 💢 Emotional Manipulation Score</strong><br>
    Counts emotionally charged words, ALL-CAPS usage, and excessive punctuation to score manipulation level.<br><br>
    <strong>4. 📰 Source Credibility</strong><br>
    Each news source has a trust score (0-100) stored in SQLite, seeded from known trusted/untrusted outlets.<br><br>
    <strong>5. 📡 Real-time Fact Check</strong><br>
    Queries NewsAPI.org with key terms to surface related verified articles.<br><br>
    <strong>6. 🗣️ Feedback Loop</strong><br>
    User feedback is stored and displayed in the History tab for continual model improvement.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">🚀 Quick Start Guide</p>', unsafe_allow_html=True)
    st.code("""# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML model
python model_trainer.py

# 3. (Optional) Set NewsAPI key for real-time fact-checking
$env:NEWS_API_KEY = "your_key_here"   # Windows PowerShell

# 4. Launch the app
streamlit run app.py""", language="bash")
