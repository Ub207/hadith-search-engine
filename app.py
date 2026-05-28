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
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DATA_DIR   = Path("data")
HF_REPO_ID = "ubaid-ai/hadith-search-engine-data"

_HF_REQUIRED = [
    "hadith_index.faiss",
    "hadith_metadata.pkl",
    "collections_meta.json",
]


def ensure_data_files() -> bool:
    if all((DATA_DIR / f).exists() for f in _HF_REQUIRED):
        return True

    DATA_DIR.mkdir(exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        st.error("Run `pip install huggingface_hub` then restart the app.")
        return False

    for filename in _HF_REQUIRED:
        if (DATA_DIR / filename).exists():
            continue
        st.info(f"⏳ Downloading **{filename}** from HuggingFace…")
        try:
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            st.error(
                f"Download failed for **{filename}**.\n\n`{exc}`\n\n"
                f"Make sure the dataset `{HF_REPO_ID}` is public and the files are uploaded."
            )
            return False
    return True


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hadith AI Search Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;700&display=swap" rel="stylesheet">
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ─── GLOBAL LIGHT THEME ─── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background-color: #FAFAFA !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}
.block-container {
    padding-top: 0 !important;
    max-width: 1200px !important;
    background: #FAFAFA !important;
}

/* ─── SIDEBAR ─── */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #374151 !important;
}

/* ─── BANNER ─── */
.app-banner {
    background: linear-gradient(160deg, #ECFDF5 0%, #F0FDF4 60%, #FFFFFF 100%);
    border-bottom: 3px solid #059669;
    padding: 2rem 2rem 1.6rem;
    margin: -1rem -1rem 2rem -1rem;
    text-align: center;
}
.bismillah-banner {
    font-family: 'Amiri', 'Traditional Arabic', 'Noto Naskh Arabic', serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #047857;
    direction: rtl;
    unicode-bidi: bidi-override;
    display: block;
    margin-bottom: 0.4rem;
    line-height: 1.8;
}
.banner-title {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #1F2937;
    letter-spacing: -0.5px;
    margin: 0.2rem 0 0.3rem;
}
.banner-subtitle { color: #6B7280; font-size: 0.93rem; margin-bottom: 0.3rem; }
.banner-byline { color: #059669; font-size: 0.82rem; font-weight: 600; letter-spacing: 0.3px; }
.banner-byline a { color: #059669; text-decoration: none; }
.banner-byline a:hover { text-decoration: underline; color: #047857; }

/* ─── TRANSLATION NOTICE ─── */
.translation-notice {
    background: #FFF7ED;
    border: 1px solid #FED7AA;
    border-left: 4px solid #F97316;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.83rem;
    color: #9A3412;
}

/* ─── SEARCH BOX ─── */
.search-wrap {
    background: #FFFFFF;
    border: 1.5px solid #D1D5DB;
    border-radius: 16px;
    padding: 1.5rem 1.8rem 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
}
.search-label {
    color: #1F2937;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}

/* ─── AI ANSWER BOX ─── */
.ai-header {
    background: #ECFDF5;
    border-radius: 10px 10px 0 0;
    padding: 0.65rem 1.2rem;
    color: #065F46;
    font-weight: 700;
    font-size: 0.88rem;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    border: 1.5px solid #059669;
    border-bottom: none;
}
.ai-body {
    background: #FFFFFF;
    border: 1.5px solid #059669;
    border-top: none;
    border-radius: 0 0 10px 10px;
    padding: 1.2rem 1.5rem 1.1rem;
    margin-bottom: 1.8rem;
    color: #1F2937 !important;
    line-height: 1.85;
    font-size: 0.94rem;
}
.ai-body p, .ai-body li { color: #1F2937 !important; }
.ai-body strong { color: #065F46 !important; }

/* ─── HADITH CARDS ─── */
.hadith-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-left: 4px solid #059669;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s ease;
}
.hadith-card:hover {
    box-shadow: 0 6px 20px rgba(5,150,105,0.1);
}
.hcard-header {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    align-items: center;
    border-bottom: 1px solid #F3F4F6;
    padding-bottom: 0.6rem;
    margin-bottom: 0.8rem;
}

/* ─── COLLECTION BADGES ─── */
.badge-bukhari  { background:#D1FAE5; color:#065F46; border:1px solid #6EE7B7; }
.badge-muslim   { background:#DBEAFE; color:#1E3A8A; border:1px solid #93C5FD; }
.badge-abudawud { background:#EDE9FE; color:#4C1D95; border:1px solid #C4B5FD; }
.badge-tirmidhi { background:#FEF3C7; color:#92400E; border:1px solid #FCD34D; }
.badge-nasai    { background:#CCFBF1; color:#134E4A; border:1px solid #5EEAD4; }
.badge-ibnmajah { background:#FEE2E2; color:#7F1D1D; border:1px solid #FCA5A5; }
.badge-malik    { background:#FDF4FF; color:#581C87; border:1px solid #D8B4FE; }
.badge-default  { background:#D1FAE5; color:#065F46; border:1px solid #059669; }

.badge-collection {
    font-size: 0.75rem; font-weight: 700;
    padding: 0.2rem 0.7rem; border-radius: 20px; white-space: nowrap;
}
.badge-ref { color: #6B7280; font-size: 0.78rem; font-weight: 600; white-space: nowrap; }

/* ─── GRADE BADGES ─── */
.grade-s {
    background:#D1FAE5; color:#065F46; border:1px solid #6EE7B7;
    font-size:0.7rem; font-weight:700; padding:0.15rem 0.55rem; border-radius:12px; white-space:nowrap;
}
.grade-h {
    background:#FEF3C7; color:#92400E; border:1px solid #FCD34D;
    font-size:0.7rem; font-weight:700; padding:0.15rem 0.55rem; border-radius:12px; white-space:nowrap;
}
.grade-d {
    background:#FEE2E2; color:#7F1D1D; border:1px solid #FCA5A5;
    font-size:0.7rem; font-weight:700; padding:0.15rem 0.55rem; border-radius:12px; white-space:nowrap;
}
.grade-n {
    background:#F3F4F6; color:#6B7280; border:1px solid #D1D5DB;
    font-size:0.7rem; padding:0.15rem 0.55rem; border-radius:12px; white-space:nowrap;
}
.graders-note { color: #9CA3AF; font-size: 0.68rem; }

/* ─── LANGUAGE BLOCKS ─── */
.lang-label {
    font-size:0.72rem; font-weight:700; color:#059669;
    text-transform:uppercase; letter-spacing:1.1px; margin:0.55rem 0 0.25rem;
}
.arabic-block {
    direction:rtl; font-family:'Amiri','Traditional Arabic','Noto Naskh Arabic',serif;
    font-size:1.45rem; color:#1F2937; line-height:2.3; text-align:right;
    background:#F0FDF4; border-left:3px solid #059669; border-radius:8px;
    padding:0.7rem 1rem; margin:0.3rem 0 0.7rem;
}
.narrator-strip {
    color:#6B7280; font-style:italic; border-left:3px solid #059669;
    padding-left:0.65rem; margin:0.2rem 0 0.45rem;
    font-size:0.84rem; line-height:1.6;
}
.english-block { color:#374151; line-height:1.85; font-size:0.93rem; margin-bottom:0.5rem; }
.urdu-block {
    direction:rtl; font-family:'Amiri','Noto Nastaliq Urdu',serif;
    font-size:1.05rem; color:#374151; line-height:2.3; text-align:right;
    background:#F9FAFB; border-radius:8px;
    padding:0.5rem 0.9rem; margin:0.3rem 0 0.4rem;
}

/* ─── SIDEBAR COMPONENTS ─── */
.sb-title {
    color:#059669; font-size:0.73rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1.3px;
    border-bottom:1.5px solid #D1FAE5; padding-bottom:0.3rem; margin-bottom:0.55rem;
}
.stat-card {
    background:#F0FDF4; border:1px solid #A7F3D0;
    border-radius:10px; padding:0.6rem 0.8rem; margin-bottom:0.6rem; text-align:center;
}
.stat-num { font-size:1.6rem; font-weight:800; color:#059669; display:block; line-height:1.2; }
.stat-lbl { color:#6B7280; font-size:0.73rem; }

.stSidebar .stButton > button {
    background:#F9FAFB !important; color:#374151 !important;
    border:1px solid #E5E7EB !important; border-radius:8px !important;
    text-align:left !important; font-size:0.8rem !important;
    padding:0.4rem 0.65rem !important; width:100% !important;
    margin-bottom:4px !important; white-space:normal !important;
    height:auto !important; line-height:1.45 !important;
}
.stSidebar .stButton > button:hover {
    background:#ECFDF5 !important;
    border-color:#059669 !important; color:#065F46 !important;
}

/* ─── FORM ELEMENTS ─── */
.stTextInput input {
    background:#FFFFFF !important;
    border:1.5px solid #D1D5DB !important;
    border-radius:10px !important;
    color:#1F2937 !important;
    font-size:1rem !important;
}
.stTextInput input:focus {
    border-color:#059669 !important;
    box-shadow:0 0 0 3px rgba(5,150,105,0.08) !important;
}
.stTextInput input::placeholder { color:#9CA3AF !important; }

button[kind="primaryFormSubmit"],
button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #34D399 0%, #10B981 100%) !important;
    border-color: #10B981 !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 8px rgba(16,185,129,0.25) !important;
    letter-spacing: 0.3px !important;
}
button[kind="primaryFormSubmit"]:hover,
button[data-testid="baseButton-primary"]:hover {
    background: linear-gradient(135deg, #6EE7B7 0%, #34D399 100%) !important;
    border-color: #34D399 !important;
    box-shadow: 0 4px 14px rgba(52,211,153,0.35) !important;
}

.stSelectbox > div > div {
    background:#FFFFFF !important;
    border:1.5px solid #D1D5DB !important;
    border-radius:8px !important;
    color:#1F2937 !important;
}

hr { border-color:#E5E7EB !important; }

/* ─── FOOTER ─── */
.footer-wrap {
    background:#FFFFFF; border:1px solid #E5E7EB;
    border-radius:12px; padding:1.3rem 1.5rem; text-align:center; margin-top:2.5rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
}
.footer-title { color:#1F2937; font-size:0.9rem; font-weight:700; margin-bottom:0.25rem; }
.footer-sub { color:#6B7280; font-size:0.78rem; margin-bottom:0.5rem; }
.footer-disclaimer { color:#9CA3AF; font-size:0.73rem; margin-bottom:0.6rem; font-style:italic; }
.footer-links a {
    color:#059669; text-decoration:none; font-size:0.8rem;
    font-weight:600; margin:0 0.5rem;
}
.footer-links a:hover { text-decoration:underline; color:#047857; }

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#F9FAFB; }
::-webkit-scrollbar-thumb { background:#D1D5DB; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#9CA3AF; }
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
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
    "Sahih al-Bukhari":   {"badge_cls": "badge-bukhari",  "icon": "📗", "key": "bukhari"},
    "Sahih Muslim":       {"badge_cls": "badge-muslim",   "icon": "📘", "key": "muslim"},
    "Sunan Abu Dawud":    {"badge_cls": "badge-abudawud", "icon": "📙", "key": "abudawud"},
    "Jami at-Tirmidhi":   {"badge_cls": "badge-tirmidhi", "icon": "📕", "key": "tirmidhi"},
    "Sunan an-Nasa'i":    {"badge_cls": "badge-nasai",    "icon": "📒", "key": "nasai"},
    "Sunan Ibn Majah":    {"badge_cls": "badge-ibnmajah", "icon": "📓", "key": "ibnmajah"},
    "Muwatta Imam Malik": {"badge_cls": "badge-malik",    "icon": "📔", "key": "malik"},
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

CRITICAL — Arabic Matan (متن عربی):
- You MUST include the original Arabic text (matan) of each relevant hadith in your answer
- Format Arabic text on its own line, right-aligned, like this:

  **Arabic (متن):**
  «حديث كا عربی متن یہاں»

- This is mandatory — every hadith reference in your answer MUST show its Arabic matan
- The Arabic text is provided in the context under "Arabic:" — copy it exactly as given
- Place the Arabic matan BEFORE the English/Urdu translation

Citation format: [Collection — Book Name — Hadith #number — Grade]

IMPORTANT:
- Never fabricate hadith or references
- Always ground answers in provided context only
- If multiple hadith on a topic, present them organized by collection
- Clearly note if a hadith is Daif (weak)
- ALWAYS include Arabic matan — this is non-negotiable\
"""

TRANSLATE_SYSTEM_PROMPT = """\
You are a search query translator.

Your ONLY job: Convert Urdu or Roman Urdu queries into concise English search queries.

Rules:
- If input is Roman Urdu (e.g. "naya kapra pehnny ki dua") → translate to English ("dua for wearing new clothes")
- If input is Urdu script (e.g. "نماز کی اہمیت") → translate to English ("importance of prayer")
- If input is already English → return it exactly as-is, no changes
- Output ONLY the translated query — no explanation, no punctuation, no extra words
- Keep it under 10 words
- Focus on the Islamic/Hadith topic being asked about

Examples:
Input: "naya kapra pehnny ki bukhari my konci dua ai h"
Output: dua for wearing new clothes

Input: "sabr ki hadith"
Output: hadith about patience

Input: "What did Prophet say about honesty"
Output: What did Prophet say about honesty

Input: "wudu ka tarika"
Output: method of performing wudu ablution\
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


@st.cache_resource(show_spinner=False)
def get_groq_client():
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ── Translation helper ─────────────────────────────────────────────────────────
def translate_query_to_english(query: str, client) -> tuple[str, bool]:
    """
    Roman Urdu / Urdu queries ko English mein translate karo for better FAISS search.
    Returns: (translated_query, was_translated)
    """
    if client is None:
        return query, False

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=30,
            temperature=0.1,
        )
        translated = resp.choices[0].message.content.strip()

        was_translated = translated.lower() != query.lower()
        return translated, was_translated

    except Exception:
        return query, False


# ── RAG helpers ────────────────────────────────────────────────────────────────
def search_hadiths(query, model, index, metadata, k=5, collection_filter=None, grade_filter="All"):
    import numpy as np
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    fetch_k = min(k * 8, len(metadata))
    _, idxs = index.search(q_emb, fetch_k)

    results = []
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


def build_context(hadiths):
    parts = []
    for i, h in enumerate(hadiths, 1):
        ref = f"[{i}] {h['collection']} — {h['book_name']} — Hadith #{h['hadith_number']}"
        if h.get("grade_summary"):
            ref += f" — Grade: {h['grade_summary']}"
        lines = [ref]
        if h.get("arabic"):
            lines.append(f"Arabic: {h['arabic']}")
        if h.get("english"):
            lines.append(f"English: {h['english']}")
        if h.get("urdu"):
            lines.append(f"Urdu: {h['urdu']}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def generate_answer(client, query, context):
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Hadith Context:\n\n{context}\n\nQuestion: {query}"},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    return resp.choices[0].message.content


# ── UI helpers ─────────────────────────────────────────────────────────────────
def _grade_badge(grade_summary):
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


def render_hadith_card(h):
    info = COLLECTION_INFO.get(h.get("collection", ""), {})
    badge_cls = info.get("badge_cls", "badge-default")
    icon = info.get("icon", "📗")
    col_name = html_module.escape(h.get("collection", ""))
    book_name = html_module.escape(h.get("book_name", "Unknown"))
    hadith_num = h.get("hadith_number", "")
    grade_html = _grade_badge(h.get("grade_summary", ""))

    graders_html = ""
    if h.get("grades"):
        parts = [
            f"{html_module.escape(g.get('name',''))}: {html_module.escape(g.get('grade',''))}"
            for g in h["grades"][:2] if g.get("grade")
        ]
        if parts:
            graders_html = f' <span class="graders-note">({", ".join(parts)})</span>'

    header = f"""<div class="hcard-header">
        <span class="badge-collection {badge_cls}">{icon} {col_name}</span>
        <span class="badge-ref">📖 {book_name} &nbsp;|&nbsp; #{hadith_num}</span>
        {grade_html}{graders_html}
    </div>"""

    arabic_html = ""
    if h.get("arabic"):
        arabic_html = f'<div class="lang-label">🕌 Arabic</div><div class="arabic-block">{html_module.escape(h["arabic"])}</div>'

    narrator_html = ""
    eng_text = h.get("english", "")
    if eng_text.startswith("Narrated") and ":" in eng_text[:200]:
        narrator, main = eng_text.split(":", 1)
        narrator_html = f'<div class="narrator-strip">⟨ {html_module.escape(narrator.strip())} ⟩</div>'
        eng_body = html_module.escape(main.strip())
    else:
        eng_body = html_module.escape(eng_text)
    eng_html = f'<div class="lang-label">🇬🇧 English</div>{narrator_html}<div class="english-block">{eng_body}</div>'

    urdu_html = ""
    if h.get("urdu"):
        urdu_html = f'<div class="lang-label">🇵🇰 Urdu</div><div class="urdu-block">{html_module.escape(h["urdu"])}</div>'

    st.markdown(
        f'<div class="hadith-card">{header}{arabic_html}{eng_html}{urdu_html}</div>',
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Guard: ensure data files present
    if not (DATA_DIR / "hadith_index.faiss").exists():
        with st.spinner("⏳ First launch: downloading Hadith database (~180 MB) — ~2 minutes…"):
            if not ensure_data_files():
                st.stop()
        st.rerun()

    # Load resources
    try:
        with st.spinner("📚 Loading hadith database…"):
            embed_model, faiss_index, metadata = load_faiss_resources()
    except Exception as e:
        st.error(f"**Failed to load database:** `{e}`\n\nRun `python prepare_data.py` to rebuild.")
        st.stop()

    groq_client = get_groq_client()

    all_collection_names = sorted({h["collection"] for h in metadata})
    collection_counts = {n: sum(1 for h in metadata if h["collection"] == n) for n in all_collection_names}
    total_hadith = len(metadata)

    # Banner
    count_str = f"{total_hadith:,}+" if total_hadith else "30,000+"
    st.markdown(
        f"""<div class="app-banner">
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

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
<div style="text-align:center;padding:1.1rem 0 0.9rem;border-bottom:1.5px solid #D1FAE5;margin-bottom:1.1rem;">
    <div style="font-size:2.4rem;line-height:1;">☪️</div>
    <div style="font-size:0.8rem;font-weight:700;color:#059669;text-transform:uppercase;
                letter-spacing:1.2px;margin-top:0.35rem;">Hadith Search</div>
    <div style="font-size:0.68rem;color:#9CA3AF;margin-top:0.1rem;">Sihah Sitta · AI-Powered</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-title">📊 Database</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="stat-card"><span class="stat-num">{total_hadith:,}</span>'
            f'<span class="stat-lbl">Total Hadith Indexed</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="stat-card"><span class="stat-num">{len(all_collection_names)}</span>'
            f'<span class="stat-lbl">Collections Available</span></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        st.markdown('<div class="sb-title">📚 Filter Collections</div>', unsafe_allow_html=True)
        selected_collections = []
        for name in all_collection_names:
            cnt = collection_counts.get(name, 0)
            icon = COLLECTION_INFO.get(name, {}).get("icon", "📗")
            if st.checkbox(f"{icon} {name} ({cnt:,})", value=True, key=f"chk_{name}"):
                selected_collections.append(name)

        st.divider()

        st.markdown('<div class="sb-title">🏅 Grade Filter</div>', unsafe_allow_html=True)
        grade_filter = st.selectbox(
            "Grade",
            ["All", "Sahih only", "Hasan", "Daif"],
            label_visibility="collapsed",
        )

        st.divider()

        st.markdown('<div class="sb-title">💡 Example Questions</div>', unsafe_allow_html=True)
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{abs(hash(q))}"):
                st.session_state["_example_q"] = q
                st.rerun()

    # ── SEARCH ────────────────────────────────────────────────────────────────
    example_q = st.session_state.pop("_example_q", None)
    form_default = example_q if example_q else st.session_state.get("_current_query", "")

    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="search-label">🔍 Ask about any Hadith in English, Urdu, or Roman Urdu</div>',
        unsafe_allow_html=True,
    )

    with st.form("search_form", clear_on_submit=False):
        typed = st.text_input(
            "query",
            value=form_default,
            placeholder="e.g. 'naya kapra pehnny ki dua' or 'Hadith about patience' or 'نماز کی اہمیت'",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([6, 1])
        with c1:
            submitted = st.form_submit_button("Ask AI 🤖", type="primary", use_container_width=True)
        with c2:
            cleared = st.form_submit_button("Clear", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle Clear
    if cleared:
        for k in ("_current_query", "_last_results", "_last_answer", "_translated_query", "_was_translated"):
            st.session_state.pop(k, None)
        st.rerun()

    # Decide what to search
    run_query = None
    if submitted and typed.strip():
        run_query = typed.strip()
        st.session_state["_current_query"] = run_query
        st.session_state.pop("_last_results", None)
        st.session_state.pop("_last_answer", None)
        st.session_state.pop("_translated_query", None)
        st.session_state.pop("_was_translated", None)
    elif example_q:
        run_query = example_q
        st.session_state["_current_query"] = run_query
        st.session_state.pop("_last_results", None)
        st.session_state.pop("_last_answer", None)
        st.session_state.pop("_translated_query", None)
        st.session_state.pop("_was_translated", None)

    # ── EXECUTE SEARCH WITH TRANSLATION ───────────────────────────────────────
    if run_query:
        col_filter = selected_collections if selected_collections else None

        with st.spinner("🔄 Processing your query…"):
            # Step 1: Translate Roman Urdu / Urdu → English for FAISS
            search_query, was_translated = translate_query_to_english(run_query, groq_client)
            st.session_state["_translated_query"] = search_query
            st.session_state["_was_translated"] = was_translated

        with st.spinner("🔍 Searching across Sihah Sitta…"):
            try:
                results = search_hadiths(
                    search_query,   # ← translated English query for FAISS
                    embed_model, faiss_index, metadata,
                    k=5,
                    collection_filter=col_filter,
                    grade_filter=grade_filter,
                )
                st.session_state["_last_results"] = results
            except Exception as e:
                st.error(f"**Search error:** `{e}`")
                st.session_state["_last_results"] = []
                results = []

        if results and groq_client:
            context = build_context(results)
            with st.spinner("🤖 Generating scholarly answer…"):
                try:
                    answer = generate_answer(groq_client, run_query, context)
                    st.session_state["_last_answer"] = answer
                except Exception as e:
                    st.session_state["_last_answer"] = None
                    err_str = str(e)
                    if "429" in err_str or "rate_limit_exceeded" in err_str:
                        import re
                        wait_match = re.search(r"try again in (\d+m[\d.]+s|\d+[\d.]+s)", err_str)
                        wait_msg = f" Try again in **{wait_match.group(1)}**." if wait_match else " Please wait a few minutes and try again."
                        st.warning(f"⏳ **Groq daily token limit reached.**{wait_msg}\n\nThe hadiths above are still shown — only the AI answer is unavailable right now.")
                    else:
                        st.error(f"**Groq API error:** `{e}`")
        else:
            if not groq_client and results:
                st.warning("⚠️ GROQ_API_KEY not set. Showing source hadith only.")
            st.session_state["_last_answer"] = None

    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    last_results    = st.session_state.get("_last_results")
    last_answer     = st.session_state.get("_last_answer")
    last_query      = st.session_state.get("_current_query", "")
    translated_q    = st.session_state.get("_translated_query", "")
    was_translated  = st.session_state.get("_was_translated", False)

    if last_results is not None:

        # Show translation notice if query was translated
        if was_translated and translated_q and translated_q.lower() != last_query.lower():
            st.markdown(
                f'<div class="translation-notice">'
                f'🔄 <strong>Query translated:</strong> "{html_module.escape(last_query)}" '
                f'→ searched as "<strong>{html_module.escape(translated_q)}</strong>" for better results'
                f'</div>',
                unsafe_allow_html=True,
            )

        if not last_results:
            st.warning(
                "No hadith found for your query and current filters.  \n"
                "Try different words, or uncheck a grade filter in the sidebar."
            )
        else:
            if last_answer:
                st.markdown('<div class="ai-header">🤖 AI Scholar\'s Answer</div>', unsafe_allow_html=True)
                st.markdown('<div class="ai-body">', unsafe_allow_html=True)
                st.markdown(last_answer)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"**📖 Source Hadith — {len(last_results)} retrieved**"
                + (f' *(query: "{last_query}")*' if last_query else "")
            )
            st.markdown('<hr style="border-color:#E5E7EB;margin:0.4rem 0 0.8rem">', unsafe_allow_html=True)
            for h in last_results:
                render_hadith_card(h)
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "👆 Type your question above and click **Ask AI 🤖**, "
            "or pick an example question from the sidebar.\n\n"
            "✅ You can ask in **English**, **Urdu (اردو)**, or **Roman Urdu** — all supported!"
        )

    # ── FOOTER ────────────────────────────────────────────────────────────────
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
