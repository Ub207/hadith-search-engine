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

DATA_DIR = Path("data")

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
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital@0;1&family=Playfair+Display:wght@600;700&display=swap');

.block-container { padding-top: 0.5rem; max-width: 1200px; }
section[data-testid="stSidebar"] { min-width: 290px; }

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
.banner-subtitle { color: #90CAF9; font-size: 0.88rem; margin-bottom: 0.25rem; }
.banner-byline { color: #FFC107; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.4px; }
.banner-byline a { color: #FFC107; text-decoration: none; }
.banner-byline a:hover { text-decoration: underline; color: #FFD54F; }
.bismillah-banner {
    font-family: 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.7rem;
    color: #FFD54F;
    direction: rtl;
    display: block;
    margin-bottom: 0.25rem;
    text-shadow: 0 1px 5px rgba(0,0,0,0.35);
}

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

.badge-bukhari  { background:#1B5E20; color:#A5D6A7; border:1px solid #388E3C; }
.badge-muslim   { background:#0D47A1; color:#90CAF9; border:1px solid #1565C0; }
.badge-abudawud { background:#4A148C; color:#CE93D8; border:1px solid #7B1FA2; }
.badge-tirmidhi { background:#E65100; color:#FFCC80; border:1px solid #F57C00; }
.badge-nasai    { background:#004D40; color:#80CBC4; border:1px solid #00695C; }
.badge-ibnmajah { background:#B71C1C; color:#FFCDD2; border:1px solid #C62828; }
.badge-malik    { background:#4E342E; color:#BCAAA4; border:1px solid #6D4C41; }
.badge-default  { background:#0D47A1; color:#FFC107; border:1px solid #FFC107; }

.badge-collection {
    font-size: 0.75rem; font-weight: 700;
    padding: 0.18rem 0.65rem; border-radius: 20px; white-space: nowrap;
}
.badge-ref { color: #90CAF9; font-size: 0.76rem; font-weight: 600; white-space: nowrap; }

.grade-s {
    background:rgba(46,125,50,0.22); color:#81C784; border:1px solid #388E3C;
    font-size:0.7rem; font-weight:700; padding:0.13rem 0.5rem; border-radius:12px; white-space:nowrap;
}
.grade-h {
    background:rgba(245,127,23,0.22); color:#FFB74D; border:1px solid #EF6C00;
    font-size:0.7rem; font-weight:700; padding:0.13rem 0.5rem; border-radius:12px; white-space:nowrap;
}
.grade-d {
    background:rgba(183,28,28,0.22); color:#EF9A9A; border:1px solid #C62828;
    font-size:0.7rem; font-weight:700; padding:0.13rem 0.5rem; border-radius:12px; white-space:nowrap;
}
.grade-n {
    background:rgba(69,90,100,0.22); color:#90A4AE; border:1px solid #546E7A;
    font-size:0.7rem; padding:0.13rem 0.5rem; border-radius:12px; white-space:nowrap;
}
.graders-note { color:#546E7A; font-size:0.68rem; }

.lang-label {
    font-size:0.7rem; font-weight:700; color:#FFC107;
    text-transform:uppercase; letter-spacing:1px; margin:0.5rem 0 0.2rem;
}
.arabic-block {
    direction:rtl; font-family:'Amiri','Traditional Arabic',serif;
    font-size:1.4rem; color:#FFD54F; line-height:2.3; text-align:right;
    background:rgba(13,71,161,0.1); border-radius:6px;
    padding:0.6rem 0.9rem; margin:0.3rem 0 0.7rem;
}
.narrator-strip {
    color:#78909C; font-style:italic; border-left:3px solid #1565C0;
    padding-left:0.65rem; margin:0.2rem 0 0.45rem;
    font-size:0.83rem; line-height:1.6;
}
.english-block { color:#E3F2FD; line-height:1.8; font-size:0.92rem; margin-bottom:0.5rem; }
.urdu-block {
    direction:rtl; font-family:'Amiri','Noto Nastaliq Urdu',serif;
    font-size:1.05rem; color:#B3E5FC; line-height:2.2; text-align:right;
    background:rgba(13,71,161,0.07); border-radius:6px;
    padding:0.4rem 0.8rem; margin:0.3rem 0 0.4rem;
}

.sb-title {
    color:#FFC107; font-size:0.78rem; font-weight:700;
    text-transform:uppercase; letter-spacing:1px;
    border-bottom:1px solid #1565C0; padding-bottom:0.3rem; margin-bottom:0.5rem;
}
.stat-card {
    background:rgba(13,71,161,0.18); border:1px solid #1565C0;
    border-radius:8px; padding:0.5rem 0.7rem; margin-bottom:0.65rem; text-align:center;
}
.stat-num { font-size:1.5rem; font-weight:700; color:#FFC107; display:block; line-height:1.2; }
.stat-lbl { color:#90CAF9; font-size:0.72rem; }

.stSidebar .stButton > button {
    background:rgba(13,71,161,0.18) !important; color:#B3E5FC !important;
    border:1px solid #1565C0 !important; border-radius:6px !important;
    text-align:left !important; font-size:0.77rem !important;
    padding:0.32rem 0.55rem !important; width:100% !important;
    margin-bottom:3px !important; white-space:normal !important;
    height:auto !important; line-height:1.4 !important;
}
.stSidebar .stButton > button:hover {
    background:rgba(13,71,161,0.48) !important;
    border-color:#FFC107 !important; color:#FFF9C4 !important;
}

.footer-wrap {
    background:rgba(13,71,161,0.10); border:1px solid #1A237E;
    border-radius:10px; padding:1rem 1.5rem; text-align:center; margin-top:2rem;
}
.footer-title { color:#90CAF9; font-size:0.85rem; font-weight:700; margin-bottom:0.2rem; }
.footer-sub { color:#546E7A; font-size:0.75rem; margin-bottom:0.6rem; }
.footer-disclaimer { color:#546E7A; font-size:0.72rem; margin-bottom:0.6rem; font-style:italic; }
.footer-links a {
    color:#FFC107; text-decoration:none; font-size:0.78rem;
    font-weight:600; margin:0 0.5rem;
}
.footer-links a:hover { text-decoration:underline; color:#FFD54F; }
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
- Always cite the exact collection name, book name, and hadith number
- Mention the hadith grade (Sahih/Hasan/Daif) when available
- Always write the Prophet's name as: Prophet Muhammad ﷺ
- Be respectful and scholarly in tone
- Answer in English or Urdu based on the user's language
- If the topic is not covered by the provided hadith, say so honestly

Citation format: [Collection — Book Name — Hadith #number — Grade]

IMPORTANT: Never fabricate hadith or references. Only use the provided context.\
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
        model="llama-3.3-70b-versatile",
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
    # Guard: data must exist
    if not (DATA_DIR / "hadith_index.faiss").exists():
        st.error("**Hadith database not found.** Run: `python prepare_data.py`")
        st.stop()

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
    # Pick up example question if sidebar button was clicked
    example_q = st.session_state.pop("_example_q", None)

    # What to show in the text box: example click → that question; otherwise last searched query
    form_default = example_q if example_q else st.session_state.get("_current_query", "")

    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
    st.markdown(
        '<div class="search-label">🔍 Ask about any Hadith in English or Urdu</div>',
        unsafe_allow_html=True,
    )

    # st.form captures button + text_input atomically — no state-loss across reruns
    with st.form("search_form", clear_on_submit=False):
        typed = st.text_input(
            "query",
            value=form_default,
            placeholder="e.g. 'Hadith about patience' or 'نماز کی اہمیت'",
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
        for k in ("_current_query", "_last_results", "_last_answer"):
            st.session_state.pop(k, None)
        st.rerun()

    # Decide what to search
    run_query = None
    if submitted and typed.strip():
        run_query = typed.strip()
        st.session_state["_current_query"] = run_query
        # Clear old results so we don't show stale data while spinner runs
        st.session_state.pop("_last_results", None)
        st.session_state.pop("_last_answer", None)
    elif example_q:
        # Sidebar example was clicked → auto-search without waiting for form submit
        run_query = example_q
        st.session_state["_current_query"] = run_query
        st.session_state.pop("_last_results", None)
        st.session_state.pop("_last_answer", None)

    # Execute search
    if run_query:
        col_filter = selected_collections if selected_collections else None

        with st.spinner("🔍 Searching across Sihah Sitta…"):
            try:
                results = search_hadiths(
                    run_query, embed_model, faiss_index, metadata,
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
                    st.error(f"**Groq API error:** `{e}`")
        else:
            if not groq_client and results:
                st.warning("⚠️ GROQ_API_KEY not set. Showing source hadith only.")
            st.session_state["_last_answer"] = None

    # ── DISPLAY RESULTS ───────────────────────────────────────────────────────
    last_results = st.session_state.get("_last_results")
    last_answer = st.session_state.get("_last_answer")
    last_query = st.session_state.get("_current_query", "")

    if last_results is not None:
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
            st.markdown('<hr style="border-color:#1565C0;margin:0.4rem 0 0.8rem">', unsafe_allow_html=True)
            for h in last_results:
                render_hadith_card(h)
    else:
        # Nothing searched yet — simple prompt
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "👆 Type your question above and click **Ask AI 🤖**, "
            "or pick an example question from the sidebar."
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
