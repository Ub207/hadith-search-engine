"""
Hadith AI Search Engine
RAG search across all Sihah Sitta (6 major Hadith collections)
Arabic · English · Urdu  ·  Groq AI  ·  FAISS semantic search
Built by Ubaid ur Rehman — Aalim | AI Developer
"""

import html as html_module
import json
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hadith AI Search Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Deep Blue & Gold Islamic Scholar Theme ───────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital@0;1&family=Playfair+Display:wght@600;700&display=swap');

/* ─ Global layout ─ */
.block-container { padding-top: 0.5rem; max-width: 1200px; }
section[data-testid="stSidebar"] { min-width: 290px; }

/* ─ Top banner ─ */
.app-banner {
    background: linear-gradient(135deg, #0A2E6F 0%, #0D47A1 55%, #1565C0 100%);
    border-bottom: 3px solid #FFC107;
    padding: 1.1rem 2rem 0.9rem;
    margin: -0.5rem -1rem 1.4rem -1rem;
    text-align: center;
}
.banner-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFF9C4;
    letter-spacing: 0.8px;
    margin: 0.25rem 0 0.1rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.45);
}
.banner-subtitle {
    color: #90CAF9;
    font-size: 0.88rem;
    margin-bottom: 0.25rem;
}
.banner-byline {
    color: #FFC107;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.4px;
}
.banner-byline a {
    color: #FFC107;
    text-decoration: none;
}
.banner-byline a:hover {
    text-decoration: underline;
    color: #FFD54F;
}
.bismillah-banner {
    font-family: 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.7rem;
    color: #FFD54F;
    direction: rtl;
    display: block;
    margin-bottom: 0.25rem;
    text-shadow: 0 1px 5px rgba(0,0,0,0.35);
    letter-spacing: 0.5px;
}

/* ─ Tabs ─ */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: 2px solid #1565C0;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(13,71,161,0.18);
    border: 1px solid #1A237E;
    border-radius: 8px 8px 0 0;
    color: #90CAF9;
    font-weight: 600;
    padding: 0.45rem 1.1rem;
    font-size: 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: #0D47A1 !important;
    color: #FFC107 !important;
    border-color: #FFC107 !important;
}

/* ─ Search bar container ─ */
.search-wrap {
    background: rgba(13,71,161,0.14);
    border: 1.5px solid #1565C0;
    border-radius: 12px;
    padding: 1.1rem 1.4rem 0.9rem;
    margin-bottom: 1rem;
}
.search-label {
    color: #FFC107;
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
}

/* ─ AI answer box ─ */
.ai-header {
    background: linear-gradient(90deg, #0D47A1, #1565C0);
    border-radius: 8px 8px 0 0;
    padding: 0.5rem 1rem;
    color: #FFC107;
    font-weight: 700;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    border: 1px solid #FFC107;
    border-bottom: none;
}
.ai-body {
    background: linear-gradient(145deg, #0A1929, #0D2137);
    border: 1px solid #FFC107;
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1.1rem 1.3rem 1rem;
    margin-bottom: 1.4rem;
    color: #E3F2FD;
    line-height: 1.85;
    font-size: 0.94rem;
}

/* ─ Hadith card ─ */
.hadith-card {
    background: linear-gradient(160deg, #0B1C3D 0%, #0A2156 100%);
    border: 1px solid #1565C0;
    border-left: 4px solid #FFC107;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 3px 14px rgba(0,0,0,0.35);
}
.hcard-header {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    border-bottom: 1px solid rgba(21,101,192,0.5);
    padding-bottom: 0.55rem;
    margin-bottom: 0.75rem;
}

/* Collection badge colours */
.badge-bukhari    { background:#1B5E20; color:#A5D6A7; border:1px solid #388E3C; }
.badge-muslim     { background:#0D47A1; color:#90CAF9; border:1px solid #1565C0; }
.badge-abudawud   { background:#4A148C; color:#CE93D8; border:1px solid #7B1FA2; }
.badge-tirmidhi   { background:#E65100; color:#FFCC80; border:1px solid #F57C00; }
.badge-nasai      { background:#004D40; color:#80CBC4; border:1px solid #00695C; }
.badge-ibnmajah   { background:#B71C1C; color:#FFCDD2; border:1px solid #C62828; }
.badge-malik      { background:#4E342E; color:#BCAAA4; border:1px solid #6D4C41; }
.badge-default    { background:#0D47A1; color:#FFC107; border:1px solid #FFC107; }

.badge-collection {
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.18rem 0.65rem;
    border-radius: 20px;
    white-space: nowrap;
}
.badge-ref {
    color: #90CAF9;
    font-size: 0.76rem;
    font-weight: 600;
    white-space: nowrap;
}
.grade-s {
    background: rgba(46,125,50,0.22);
    color: #81C784;
    border: 1px solid #388E3C;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.13rem 0.5rem;
    border-radius: 12px;
    white-space: nowrap;
}
.grade-h {
    background: rgba(245,127,23,0.22);
    color: #FFB74D;
    border: 1px solid #EF6C00;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.13rem 0.5rem;
    border-radius: 12px;
    white-space: nowrap;
}
.grade-d {
    background: rgba(183,28,28,0.22);
    color: #EF9A9A;
    border: 1px solid #C62828;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.13rem 0.5rem;
    border-radius: 12px;
    white-space: nowrap;
}
.grade-n {
    background: rgba(69,90,100,0.22);
    color: #90A4AE;
    border: 1px solid #546E7A;
    font-size: 0.7rem;
    padding: 0.13rem 0.5rem;
    border-radius: 12px;
    white-space: nowrap;
}
.graders-note {
    color: #546E7A;
    font-size: 0.68rem;
}
.lang-label {
    font-size: 0.7rem;
    font-weight: 700;
    color: #FFC107;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0.5rem 0 0.2rem;
}
.arabic-block {
    direction: rtl;
    font-family: 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.4rem;
    color: #FFD54F;
    line-height: 2.3;
    text-align: right;
    background: rgba(13,71,161,0.1);
    border-radius: 6px;
    padding: 0.6rem 0.9rem;
    margin: 0.3rem 0 0.7rem;
}
.narrator-strip {
    color: #78909C;
    font-style: italic;
    border-left: 3px solid #1565C0;
    padding-left: 0.65rem;
    margin: 0.2rem 0 0.45rem;
    font-size: 0.83rem;
    line-height: 1.6;
}
.english-block {
    color: #E3F2FD;
    line-height: 1.8;
    font-size: 0.92rem;
    margin-bottom: 0.5rem;
}
.urdu-block {
    direction: rtl;
    font-family: 'Amiri', 'Noto Nastaliq Urdu', serif;
    font-size: 1.05rem;
    color: #B3E5FC;
    line-height: 2.2;
    text-align: right;
    background: rgba(13,71,161,0.07);
    border-radius: 6px;
    padding: 0.4rem 0.8rem;
    margin: 0.3rem 0 0.4rem;
}

/* ─ Sidebar ─ */
.sb-title {
    color: #FFC107;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-bottom: 1px solid #1565C0;
    padding-bottom: 0.3rem;
    margin-bottom: 0.5rem;
}
.stat-card {
    background: rgba(13,71,161,0.18);
    border: 1px solid #1565C0;
    border-radius: 8px;
    padding: 0.5rem 0.7rem;
    margin-bottom: 0.65rem;
    text-align: center;
}
.stat-num {
    font-size: 1.5rem;
    font-weight: 700;
    color: #FFC107;
    display: block;
    line-height: 1.2;
}
.stat-lbl {
    color: #90CAF9;
    font-size: 0.72rem;
}

/* ─ Sidebar example buttons ─ */
.stSidebar .stButton > button {
    background: rgba(13,71,161,0.18) !important;
    color: #B3E5FC !important;
    border: 1px solid #1565C0 !important;
    border-radius: 6px !important;
    text-align: left !important;
    font-size: 0.77rem !important;
    padding: 0.32rem 0.55rem !important;
    width: 100% !important;
    margin-bottom: 3px !important;
    white-space: normal !important;
    height: auto !important;
    line-height: 1.4 !important;
}
.stSidebar .stButton > button:hover {
    background: rgba(13,71,161,0.48) !important;
    border-color: #FFC107 !important;
    color: #FFF9C4 !important;
}

/* ─ Welcome / feature cards ─ */
.welcome-card {
    background: rgba(13,71,161,0.14);
    border: 1px solid #1565C0;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    height: 100%;
}
.welcome-icon { font-size: 1.9rem; }
.welcome-title { color: #FFC107; font-weight: 700; margin: 0.3rem 0 0.15rem; }
.welcome-desc { color: #90CAF9; font-size: 0.78rem; }

/* ─ Browse collection cards ─ */
.col-card {
    background: linear-gradient(145deg, #0B1C3D, #0A2156);
    border: 1px solid #1565C0;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
    text-align: center;
}
.col-card-icon { font-size: 1.6rem; }
.col-card-name { color: #FFC107; font-weight: 700; font-size: 0.95rem; margin: 0.3rem 0 0.1rem; }
.col-card-author { color: #90CAF9; font-size: 0.75rem; margin-bottom: 0.2rem; }
.col-card-count { color: #FFC107; font-size: 1.1rem; font-weight: 700; }

/* ─ Footer ─ */
.footer-wrap {
    background: rgba(13,71,161,0.10);
    border: 1px solid #1A237E;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    text-align: center;
    margin-top: 2rem;
}
.footer-title {
    color: #90CAF9;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
.footer-sub {
    color: #546E7A;
    font-size: 0.75rem;
    margin-bottom: 0.6rem;
}
.footer-disclaimer {
    color: #546E7A;
    font-size: 0.72rem;
    margin-bottom: 0.6rem;
    font-style: italic;
}
.footer-links a {
    color: #FFC107;
    text-decoration: none;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 0 0.5rem;
}
.footer-links a:hover {
    text-decoration: underline;
    color: #FFD54F;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
EXAMPLE_QUESTIONS = [
    "What did the Prophet ﷺ say about patience?",
    "نماز کی اہمیت کے بارے میں حدیث",
    "Hadith about kindness to parents",
    "صبر اور شکر کی احادیث",
    "What are the signs of a hypocrite?",
    "علم کی فضیلت",
    "Hadith about Friday (Jumu'ah)",
    "رمضان اور روزے کی فضیلت",
]

COLLECTION_INFO = {
    "Sahih al-Bukhari":  {"badge_cls": "badge-bukhari",  "icon": "📗", "key": "bukhari",  "author": "Imam Muhammad ibn Ismail al-Bukhari (رحمه الله)"},
    "Sahih Muslim":      {"badge_cls": "badge-muslim",   "icon": "📘", "key": "muslim",   "author": "Imam Muslim ibn al-Hajjaj (رحمه الله)"},
    "Sunan Abu Dawud":   {"badge_cls": "badge-abudawud", "icon": "📙", "key": "abudawud", "author": "Imam Abu Dawud Sulayman (رحمه الله)"},
    "Jami at-Tirmidhi":  {"badge_cls": "badge-tirmidhi", "icon": "📕", "key": "tirmidhi", "author": "Imam Muhammad ibn Isa at-Tirmidhi (رحمه الله)"},
    "Sunan an-Nasa'i":   {"badge_cls": "badge-nasai",    "icon": "📒", "key": "nasai",    "author": "Imam Ahmad ibn Shu'ayb an-Nasa'i (رحمه الله)"},
    "Sunan Ibn Majah":   {"badge_cls": "badge-ibnmajah", "icon": "📓", "key": "ibnmajah", "author": "Imam Muhammad ibn Yazid Ibn Majah (رحمه الله)"},
    "Muwatta Imam Malik":{"badge_cls": "badge-malik",    "icon": "📔", "key": "malik",    "author": "Imam Malik ibn Anas (رحمه الله)"},
}

SYSTEM_PROMPT = """\
You are a Hadith AI Assistant created by Ubaid ur Rehman, an Aalim (Islamic Scholar) and AI Developer.

Your role:
- Answer questions about Hadith using ONLY the provided hadith context
- Always cite the exact collection name (e.g., Sahih Bukhari), book name, and hadith number
- Mention the hadith grade (Sahih/Hasan/Daif) when available
- Include the narrator (sanad) when present in the text
- Always write the Prophet's name as: Prophet Muhammad ﷺ
- Be respectful and scholarly in tone
- Answer in English or Urdu based on the user's language
- If asked about topics not in the provided hadith, say so honestly

Citation format: [Collection — Book Name — Hadith #number — Grade]

IMPORTANT:
- Never fabricate hadith or references
- Always ground answers in provided context only
- If multiple hadith on a topic, present them organized by collection
- Clearly note if a hadith is Daif (weak)\
"""


# ── Resource loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_faiss_resources():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(DATA_DIR / "hadith_index.faiss"))
    with open(DATA_DIR / "hadith_metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    return model, index, metadata


@st.cache_data(show_spinner=False)
def load_collections_meta() -> dict:
    meta_path = DATA_DIR / "collections_meta.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def get_groq_client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ── RAG helpers ────────────────────────────────────────────────────────────────
def search_hadiths(
    query: str,
    model,
    index,
    metadata: list,
    k: int = 5,
    collection_filter: list | None = None,
    grade_filter: str = "All",
) -> list:
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    fetch_k = min(k * 8, len(metadata))
    _, idxs = index.search(q_emb, fetch_k)

    results: list = []
    for i in idxs[0]:
        if not (0 <= i < len(metadata)):
            continue
        h = metadata[i]
        if collection_filter and h.get("collection") not in collection_filter:
            continue
        if grade_filter and grade_filter != "All":
            gs = h.get("grade_summary", "")
            if grade_filter == "Sahih only" and gs != "Sahih":
                continue
            if grade_filter == "Hasan" and gs != "Hasan":
                continue
            if grade_filter == "Daif" and gs != "Daif":
                continue
        results.append(h)
        if len(results) >= k:
            break

    return results


def build_context(hadiths: list) -> str:
    parts: list = []
    for i, h in enumerate(hadiths, 1):
        ref_line = (
            f"[{i}] {h['collection']} — {h['book_name']} — Hadith #{h['hadith_number']}"
        )
        if h.get("grade_summary"):
            ref_line += f" — Grade: {h['grade_summary']}"
        lines = [ref_line]
        if h.get("arabic"):
            lines.append(f"Arabic: {h['arabic']}")
        lines.append(f"English: {h['english']}")
        if h.get("urdu"):
            lines.append(f"Urdu: {h['urdu']}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def generate_answer(client, query: str, context: str) -> str:
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Hadith Context:\n\n{context}\n\nQuestion: {query}",
            },
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── UI components ──────────────────────────────────────────────────────────────
def _collection_badge_cls(collection_name: str) -> str:
    info = COLLECTION_INFO.get(collection_name)
    return info["badge_cls"] if info else "badge-default"


def _collection_icon(collection_name: str) -> str:
    info = COLLECTION_INFO.get(collection_name)
    return info["icon"] if info else "📗"


def _grade_badge(grade_summary: str) -> str:
    g = (grade_summary or "").strip()
    if g == "Sahih":
        return '<span class="grade-s">✅ Sahih</span>'
    if g == "Hasan":
        return '<span class="grade-h">🟡 Hasan</span>'
    if g in ("Daif", "Da'if", "Dha'if"):
        return '<span class="grade-d">⚠️ Daif</span>'
    if g:
        return f'<span class="grade-n">{html_module.escape(g)}</span>'
    return '<span class="grade-n">Grade N/A</span>'


def render_hadith_card(
    h: dict,
    show_arabic: bool = True,
    show_english: bool = True,
    show_urdu: bool = True,
) -> None:
    grade_html = _grade_badge(h.get("grade_summary", ""))

    graders_html = ""
    if h.get("grades"):
        parts = [
            f"{html_module.escape(g.get('name', ''))}: {html_module.escape(g.get('grade', ''))}"
            for g in h["grades"][:2]
            if g.get("grade")
        ]
        if parts:
            graders_html = f' <span class="graders-note">({", ".join(parts)})</span>'

    col_name = h.get("collection", "")
    badge_cls = _collection_badge_cls(col_name)
    icon = _collection_icon(col_name)
    col_name_esc = html_module.escape(col_name)
    book_name = html_module.escape(h.get("book_name", "Unknown"))
    hadith_num = h.get("hadith_number", "")

    header_html = f"""
    <div class="hcard-header">
        <span class="badge-collection {badge_cls}">{icon} {col_name_esc}</span>
        <span class="badge-ref">📖 {book_name} &nbsp;|&nbsp; #{hadith_num}</span>
        {grade_html}{graders_html}
    </div>"""

    arabic_html = ""
    if show_arabic and h.get("arabic"):
        ara = html_module.escape(h["arabic"])
        arabic_html = f"""
        <div class="lang-label">🕌 Arabic</div>
        <div class="arabic-block">{ara}</div>"""

    eng_html = ""
    if show_english or not h.get("urdu"):
        eng_text = h.get("english", "")
        narrator_html = ""
        if eng_text.startswith("Narrated") and ":" in eng_text[:200]:
            narrator, main = eng_text.split(":", 1)
            narrator_html = (
                f'<div class="narrator-strip">⟨ {html_module.escape(narrator.strip())} ⟩</div>'
            )
            body = html_module.escape(main.strip())
        else:
            body = html_module.escape(eng_text)
        eng_html = f"""
        <div class="lang-label">🇬🇧 English</div>
        {narrator_html}
        <div class="english-block">{body}</div>"""

    urdu_html = ""
    if show_urdu and h.get("urdu"):
        urd = html_module.escape(h["urdu"])
        urdu_html = f"""
        <div class="lang-label">🇵🇰 Urdu</div>
        <div class="urdu-block">{urd}</div>"""

    st.markdown(
        f'<div class="hadith-card">{header_html}{arabic_html}{eng_html}{urdu_html}</div>',
        unsafe_allow_html=True,
    )


def render_top_banner(total_count: int = 0) -> None:
    count_str = f"{total_count:,}+" if total_count else "30,000+"
    st.markdown(
        f"""
<div class="app-banner">
    <span class="bismillah-banner">بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</span>
    <div class="banner-title">📚 Hadith AI Search Engine</div>
    <div class="banner-subtitle">Search {count_str} authenticated Hadith across Sihah Sitta — The Six Major Collections</div>
    <div class="banner-byline">
        Built by <a href="https://ubaid-portfolios-depp.vercel.app/" target="_blank">Ubaid ur Rehman</a>
        &mdash; Aalim &amp; AI Developer
    </div>
</div>""",
        unsafe_allow_html=True,
    )


# ── Main app ───────────────────────────────────────────────────────────────────
def main() -> None:
    # Guard: FAISS index must exist
    if not (DATA_DIR / "hadith_index.faiss").exists():
        st.error("**Hadith database not found.**")
        st.markdown(
            "Run the data preparation script to download all collections and build the index:\n"
            "```bash\n"
            "python prepare_data.py\n"
            "```"
        )
        st.stop()

    # Load resources with error handling
    try:
        with st.spinner("📚 Loading hadith database…"):
            embed_model, faiss_index, metadata = load_faiss_resources()
    except Exception as e:
        st.error(
            f"**Failed to load hadith database.**\n\n"
            f"Error: `{e}`\n\n"
            "Try deleting `data/hadith_index.faiss` and `data/hadith_metadata.pkl`, "
            "then run `python prepare_data.py` again."
        )
        st.stop()

    collections_meta = load_collections_meta()
    groq_client = get_groq_client()

    all_collection_names: list = sorted({h["collection"] for h in metadata})
    collection_counts: dict = {
        name: sum(1 for h in metadata if h["collection"] == name)
        for name in all_collection_names
    }
    total_hadith = len(metadata)

    render_top_banner(total_hadith)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div class="sb-title">📊 Database</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-card">'
            f'<span class="stat-num">{total_hadith:,}</span>'
            f'<span class="stat-lbl">Total Hadith Indexed</span>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-card">'
            f'<span class="stat-num">{len(all_collection_names)}</span>'
            f'<span class="stat-lbl">Collections Available</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Collection filter ─────────────────────────────────────────────────
        st.markdown('<div class="sb-title">📚 Filter Collections</div>', unsafe_allow_html=True)
        selected_collections: list = []
        for name in all_collection_names:
            cnt = collection_counts.get(name, 0)
            info = COLLECTION_INFO.get(name, {})
            icon = info.get("icon", "📗")
            if st.checkbox(f"{icon} {name} ({cnt:,})", value=True, key=f"chk_{name}"):
                selected_collections.append(name)

        st.divider()

        # ── Grade filter ──────────────────────────────────────────────────────
        st.markdown('<div class="sb-title">🏅 Grade Filter</div>', unsafe_allow_html=True)
        grade_filter = st.selectbox(
            "Authenticity",
            ["All", "Sahih only", "Hasan", "Daif"],
            label_visibility="collapsed",
        )

        st.divider()

        # ── Language display ──────────────────────────────────────────────────
        st.markdown('<div class="sb-title">🌐 Display Language</div>', unsafe_allow_html=True)
        lang_mode = st.radio(
            "Translations",
            ["English + Urdu", "English only", "Urdu only"],
            label_visibility="collapsed",
        )
        show_arabic_cb = st.checkbox("Show Arabic text", value=True)

        st.divider()

        # ── Search settings ───────────────────────────────────────────────────
        st.markdown('<div class="sb-title">⚙️ Results Count</div>', unsafe_allow_html=True)
        k_results = st.slider("Results per search", min_value=3, max_value=10, value=5)

        st.divider()

        # ── Example questions ─────────────────────────────────────────────────
        st.markdown('<div class="sb-title">💡 Example Questions</div>', unsafe_allow_html=True)
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{abs(hash(q))}"):
                st.session_state["_pending_query"] = q
                st.rerun()

        st.divider()

        # ── About ─────────────────────────────────────────────────────────────
        st.markdown('<div class="sb-title">ℹ️ About</div>', unsafe_allow_html=True)
        st.markdown(
            "Built by **Ubaid ur Rehman**  \n"
            "Aalim | Qari | AI Developer  \n"
            "Karachi, Pakistan"
        )
        st.markdown(
            "llama-3.3-70b-versatile · FAISS · all-MiniLM-L6-v2",
            help="Model and technology stack",
        )

    # Resolve language display flags
    show_english = lang_mode in ("English + Urdu", "English only")
    show_urdu = lang_mode in ("English + Urdu", "Urdu only")

    # ── MAIN TABS ──────────────────────────────────────────────────────────────
    tab_search, tab_browse = st.tabs(["🔍 Search & Ask", "📖 Browse Collections"])

    # ── TAB 1: SEARCH & ASK ───────────────────────────────────────────────────
    with tab_search:
        # Pre-fill from example question buttons
        pending = st.session_state.pop("_pending_query", None)
        if pending:
            st.session_state["search_input"] = pending
            st.session_state["_trigger_search"] = True

        st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
        st.markdown(
            '<div class="search-label">🔍 Ask about any Hadith in English or Urdu</div>',
            unsafe_allow_html=True,
        )

        col_input, col_btn, col_clear = st.columns([5, 1, 1])
        with col_input:
            user_input = st.text_input(
                "query",
                placeholder="e.g. 'Hadith about patience' or 'نماز کی اہمیت'",
                label_visibility="collapsed",
                key="search_input",
            )
        with col_btn:
            search_clicked = st.button(
                "Ask AI 🤖", type="primary", use_container_width=True
            )
        with col_clear:
            clear_clicked = st.button("Clear", use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        if clear_clicked:
            st.session_state.pop("search_input", None)
            st.session_state.pop("_last_query", None)
            st.session_state.pop("_last_results", None)
            st.session_state.pop("_last_answer", None)
            st.session_state.pop("_trigger_search", None)
            st.rerun()

        trigger_search = st.session_state.pop("_trigger_search", False)
        query_to_run: str | None = None

        if (search_clicked or trigger_search) and user_input and user_input.strip():
            query_to_run = user_input.strip()
            st.session_state["_last_query"] = query_to_run
            st.session_state.pop("_last_results", None)
            st.session_state.pop("_last_answer", None)

        # Run search if triggered
        if query_to_run:
            col_filter = selected_collections if selected_collections else None

            with st.spinner("🔍 Searching across Sihah Sitta…"):
                try:
                    results = search_hadiths(
                        query_to_run,
                        embed_model,
                        faiss_index,
                        metadata,
                        k=k_results,
                        collection_filter=col_filter,
                        grade_filter=grade_filter,
                    )
                    st.session_state["_last_results"] = results
                except Exception as e:
                    st.error(f"**Search error:** `{e}`")
                    results = []
                    st.session_state["_last_results"] = []

            if results and groq_client:
                with st.spinner("🤖 Generating scholarly answer…"):
                    try:
                        context = build_context(results)
                        answer = generate_answer(groq_client, query_to_run, context)
                        st.session_state["_last_answer"] = answer
                    except Exception as e:
                        st.session_state["_last_answer"] = None
                        st.error(f"**Groq API error:** `{e}`")

        # Display stored results
        last_query = st.session_state.get("_last_query")
        last_results = st.session_state.get("_last_results")
        last_answer = st.session_state.get("_last_answer")

        if last_results is not None:
            # Results exist — display them
            if not last_results:
                st.warning(
                    "No hadith found matching your filters.  \n"
                    "Try expanding the collection selection or changing the grade filter."
                )
            else:
                if groq_client is None:
                    st.warning(
                        "⚠️ **GROQ_API_KEY not configured.** Add your key to `.env`.  \n"
                        "Get a free key → [console.groq.com](https://console.groq.com)"
                    )

                if last_answer:
                    st.markdown(
                        '<div class="ai-header">🤖 AI Scholar\'s Answer</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div class="ai-body">', unsafe_allow_html=True
                    )
                    st.markdown(last_answer)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown(
                    f"**📖 Source Hadith — {len(last_results)} retrieved** "
                    f"*(for: \"{last_query}\")*"
                )
                st.markdown(
                    '<hr style="border-color:#1565C0;margin:0.4rem 0 0.8rem">',
                    unsafe_allow_html=True,
                )
                for h in last_results:
                    render_hadith_card(
                        h,
                        show_arabic=show_arabic_cb,
                        show_english=show_english,
                        show_urdu=show_urdu,
                    )

        elif not last_query:
            # Welcome state — feature cards
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="welcome-card">'
                    f'<div class="welcome-icon">📚</div>'
                    f'<div class="welcome-title">{total_hadith:,}+ Hadith</div>'
                    f'<div class="welcome-desc">All Sihah Sitta collections indexed</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    '<div class="welcome-card">'
                    '<div class="welcome-icon">🔍</div>'
                    '<div class="welcome-title">Semantic Search</div>'
                    '<div class="welcome-desc">AI understands meaning — ask naturally</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    '<div class="welcome-card">'
                    '<div class="welcome-icon">🌐</div>'
                    '<div class="welcome-title">Arabic · English · Urdu</div>'
                    '<div class="welcome-desc">Three languages for every hadith</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    '<div class="welcome-card">'
                    '<div class="welcome-icon">✅</div>'
                    '<div class="welcome-title">Graded Hadith</div>'
                    '<div class="welcome-desc">Sahih · Hasan · Daif status shown</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                "👆 Type your question above and click **Ask AI**, "
                "or click an example from the sidebar."
            )

    # ── TAB 2: BROWSE COLLECTIONS ─────────────────────────────────────────────
    with tab_browse:
        # Collection overview cards
        st.markdown("### 📚 Sihah Sitta — The Six Major Collections")
        sihah_sitta_names = [
            "Sahih al-Bukhari", "Sahih Muslim", "Sunan Abu Dawud",
            "Jami at-Tirmidhi", "Sunan an-Nasa'i", "Sunan Ibn Majah",
        ]
        available_for_browse = [n for n in sihah_sitta_names if n in all_collection_names]
        other_collections = [n for n in all_collection_names if n not in sihah_sitta_names]
        if other_collections:
            available_for_browse += other_collections

        # Show 2-column collection cards
        card_cols = st.columns(2)
        for idx, col_name in enumerate(available_for_browse[:6]):
            info = COLLECTION_INFO.get(col_name, {})
            icon = info.get("icon", "📗")
            author = info.get("author", "")
            count = collection_counts.get(col_name, 0)
            with card_cols[idx % 2]:
                st.markdown(
                    f'<div class="col-card">'
                    f'<div class="col-card-icon">{icon}</div>'
                    f'<div class="col-card-name">{html_module.escape(col_name)}</div>'
                    f'<div class="col-card-author">{html_module.escape(author)}</div>'
                    f'<div class="col-card-count">{count:,} Hadith</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.divider()
        st.markdown("### 📖 Browse Hadith")

        browse_col_name = st.selectbox(
            "Select Collection",
            available_for_browse,
            key="browse_collection",
        )

        if browse_col_name:
            col_hadiths = [h for h in metadata if h["collection"] == browse_col_name]

            books: dict = {}
            for h in col_hadiths:
                bn = h.get("book_number", "")
                bname = h.get("book_name", f"Book {bn}")
                books[bn] = bname

            def _book_sort_key(item):
                k = item[0]
                try:
                    return (0, int(k))
                except (ValueError, TypeError):
                    return (1, str(k))

            sorted_books = sorted(books.items(), key=_book_sort_key)
            book_labels = [f"Book {k}: {v}" for k, v in sorted_books]
            book_keys = [k for k, _ in sorted_books]

            b_col1, b_col2, b_col3 = st.columns([3, 1, 1])
            with b_col1:
                selected_book_label = st.selectbox(
                    "Select Book / Chapter",
                    book_labels,
                    key="browse_book",
                )
            with b_col2:
                jump_to = st.number_input(
                    "Jump to Hadith #",
                    min_value=1,
                    value=1,
                    step=1,
                    key="browse_jump",
                )
            with b_col3:
                per_page = st.selectbox("Per page", [10, 20, 50], key="browse_per_page")

            sel_idx = (
                book_labels.index(selected_book_label)
                if selected_book_label in book_labels
                else 0
            )
            sel_book_key = book_keys[sel_idx] if book_keys else None

            if sel_book_key is not None:
                book_hadiths = [
                    h for h in col_hadiths
                    if str(h.get("book_number", "")) == str(sel_book_key)
                ]
            else:
                book_hadiths = col_hadiths

            book_hadiths.sort(key=lambda x: x.get("hadith_number", 0))

            st.markdown(
                f"**{browse_col_name}** — {selected_book_label} — "
                f"**{len(book_hadiths):,}** hadith"
            )
            st.markdown(
                '<hr style="border-color:#1565C0;margin:0.3rem 0 0.7rem">',
                unsafe_allow_html=True,
            )

            start_idx = 0
            for i, h in enumerate(book_hadiths):
                if h.get("hadith_number", 0) >= jump_to:
                    start_idx = i
                    break

            page_hadiths = book_hadiths[start_idx : start_idx + per_page]

            if not page_hadiths:
                st.info("No hadith found for this selection.")
            else:
                for h in page_hadiths:
                    render_hadith_card(
                        h,
                        show_arabic=show_arabic_cb,
                        show_english=show_english,
                        show_urdu=show_urdu,
                    )

                if start_idx + per_page < len(book_hadiths):
                    next_num = book_hadiths[start_idx + per_page].get("hadith_number", "")
                    st.info(
                        f"Showing hadith "
                        f"{book_hadiths[start_idx].get('hadith_number', '')}–"
                        f"{book_hadiths[min(start_idx + per_page - 1, len(book_hadiths)-1)].get('hadith_number', '')}.  "
                        f"Set **Jump to Hadith #** to **{next_num}** for the next page."
                    )

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
<div class="footer-wrap">
    <div class="footer-title">📚 Hadith AI Search Engine</div>
    <div class="footer-sub">
        Built with ❤️ by <strong>Ubaid ur Rehman</strong>
        &nbsp;—&nbsp; Aalim | Qari (Saba Qiraat) | AI Developer | Karachi, Pakistan
    </div>
    <div class="footer-disclaimer">
        ⚠️ This is an AI-powered educational tool. For authentic Islamic rulings (fatawa),
        always consult qualified scholars.
    </div>
    <div class="footer-links">
        <a href="https://ubaid-portfolios-depp.vercel.app/" target="_blank">🌐 Portfolio</a>
        <a href="https://github.com/Ub207" target="_blank">💻 GitHub</a>
        <a href="https://linkedin.com/in/ubaidurrehman-usman" target="_blank">🔗 LinkedIn</a>
        <a href="https://wa.me/923170203221" target="_blank">📱 WhatsApp</a>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
