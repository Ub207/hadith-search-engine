"""
Hadith AI Search Engine
RAG chatbot for Sahih Bukhari & Sahih Muslim — English & Urdu
Built by Ubaid ur Rehman — Aalim | AI Developer
github.com/Ub207/hadith-ai-search
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Hadith AI Search Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Deep Blue Islamic Theme ───────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:ital@0;1&display=swap');

/* ─ Layout ─ */
.block-container { padding-top: 1.5rem; }

/* ─ Header ─ */
.main-header {
    text-align: center;
    color: #E3F2FD;
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0.15rem;
    letter-spacing: 0.5px;
}
.sub-header {
    text-align: center;
    color: #90CAF9;
    font-size: 0.92rem;
    margin-bottom: 0.8rem;
}
.bismillah {
    text-align: center;
    font-family: 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.6rem;
    color: #FFD54F;
    direction: rtl;
    margin-bottom: 0.5rem;
}

/* ─ Disclaimer ─ */
.disclaimer {
    background: rgba(57, 73, 171, 0.18);
    border: 1px solid #3949AB;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #90CAF9;
    font-size: 0.78rem;
    text-align: center;
    margin-bottom: 1.2rem;
}

/* ─ Hadith card elements ─ */
.arabic-text {
    direction: rtl;
    font-family: 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.2rem;
    color: #FFD54F;
    line-height: 2.2;
    text-align: right;
    padding: 0.4rem 0;
}
.hadith-meta {
    color: #90CAF9;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.3rem 0 0;
}
.narrator {
    color: #B0BEC5;
    font-style: italic;
    border-left: 3px solid #3949AB;
    padding-left: 0.7rem;
    margin: 0.4rem 0;
    font-size: 0.88rem;
    line-height: 1.5;
}
.hadith-text {
    color: #E3F2FD;
    line-height: 1.8;
    font-size: 0.95rem;
}

/* ─ Sidebar buttons (example questions) ─ */
.stSidebar .stButton > button {
    background: rgba(57, 73, 171, 0.3);
    color: #E3F2FD;
    border: 1px solid #3949AB;
    border-radius: 6px;
    text-align: left;
    font-size: 0.8rem;
    padding: 0.4rem 0.6rem;
    width: 100%;
    margin-bottom: 2px;
    white-space: normal;
    height: auto;
}
.stSidebar .stButton > button:hover {
    background: rgba(57, 73, 171, 0.6);
    border-color: #5C6BC0;
}

/* ─ Credit ─ */
.credit {
    color: #546E7A;
    font-size: 0.73rem;
    text-align: center;
    padding: 0.4rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")

EXAMPLE_QUESTIONS = [
    "What did the Prophet (صلى الله عليه وسلم) say about kindness?",
    "Hadith about namaz/salah",
    "What is the hadith about intention (niyyah)?",
    "Sabr ke baare mein hadith",
    "Hadith about parents",
    "What did the Prophet say about knowledge?",
]

SYSTEM_PROMPT = """\
You are a Hadith AI Assistant created by Ubaid ur Rehman, an Aalim and AI Developer.

Answer questions ONLY from the provided hadith context. Always cite the exact collection \
(Bukhari/Muslim), book name, and hadith number.

Rules:
- Be scholarly and respectful in tone.
- Always write the Prophet's name as: Prophet Muhammad (صلى الله عليه وسلم)
- If the user's question is in Urdu, respond fully in Urdu.
- If in English, respond in English.
- Cite sources as: [Sahih Bukhari/Muslim — Book Name — Hadith #number]
- If the context does not contain relevant hadiths, say so honestly.
- Never fabricate, paraphrase, or invent hadith references.
- Keep answers focused, precise, and well-structured.\
"""


# ── Resource loading (cached across reruns) ───────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
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
        st.error(
            "⚠️ **GROQ_API_KEY** not found.  \n"
            "Add it to `.env` (local) or Streamlit Cloud **Secrets**."
        )
        st.stop()
    return Groq(api_key=api_key)


# ── Core RAG functions ────────────────────────────────────────────────────────
def search_hadiths(query: str, model, index, metadata, k: int = 5) -> list[dict]:
    q_emb = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    _, idxs = index.search(q_emb, k)
    return [metadata[i] for i in idxs[0] if 0 <= i < len(metadata)]


def build_context(hadiths: list[dict]) -> str:
    parts: list[str] = []
    for i, h in enumerate(hadiths, 1):
        lines = [
            f"[{i}] {h['collection']} — {h['book_name']} — Hadith #{h['hadith_number']}"
        ]
        if h.get("arabic_text"):
            lines.append(f"Arabic: {h['arabic_text']}")
        lines.append(f"English: {h['text']}")
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def generate_answer(client, query: str, context: str) -> str:
    response = client.chat.completions.create(
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
    return response.choices[0].message.content


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_hadith_expander(h: dict) -> None:
    label = f"📚 {h['collection']} — {h['book_name']} — Hadith #{h['hadith_number']}"
    with st.expander(label):
        # Arabic text (RTL)
        if h.get("arabic_text"):
            st.markdown(
                f'<div class="arabic-text">{h["arabic_text"]}</div>',
                unsafe_allow_html=True,
            )
            st.divider()

        # Narrator chain + main text
        text: str = h["text"]
        if text.startswith("Narrated") and ":" in text[:160]:
            narrator, main = text.split(":", 1)
            st.markdown(
                f'<div class="narrator">⟨ {narrator.strip()} ⟩</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="hadith-text">{main.strip()}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="hadith-text">{text}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="hadith-meta">'
            f"Collection: {h['collection']} &nbsp;|&nbsp; "
            f"Book: {h['book_name']} &nbsp;|&nbsp; "
            f"Hadith #: {h['hadith_number']}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    # Header
    st.markdown('<div class="bismillah">بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-header">📚 Hadith AI Search Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        "Sahih Bukhari &amp; Sahih Muslim &nbsp;·&nbsp; English &amp; Urdu &nbsp;·&nbsp; Powered by Groq AI"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ This tool is for educational purposes only. "
        "For authentic rulings, consult qualified scholars."
        "</div>",
        unsafe_allow_html=True,
    )

    # Guard: check data exists
    if not (DATA_DIR / "hadith_index.faiss").exists():
        st.error(
            "**Hadith database not found.**  \n"
            "Please run `python prepare_data.py` first to download and index the hadiths."
        )
        st.code("python prepare_data.py", language="bash")
        st.stop()

    # Load resources
    with st.spinner("Loading hadith database…"):
        model, index, metadata = load_resources()
    client = get_groq_client()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📚 Hadith AI")
        st.divider()

        bukhari_n = sum(1 for h in metadata if "Bukhari" in h.get("collection", ""))
        muslim_n = len(metadata) - bukhari_n

        st.markdown("**📊 Database Stats**")
        st.markdown(f"- Sahih Bukhari: **{bukhari_n:,}**")
        st.markdown(f"- Sahih Muslim: **{muslim_n:,}**")
        st.markdown(f"- Total Hadiths: **{len(metadata):,}**")
        st.divider()

        st.markdown("**💡 Example Questions**")
        for q in EXAMPLE_QUESTIONS:
            if st.button(q, key=f"ex_{hash(q)}"):
                st.session_state["_pending_query"] = q

        st.divider()

        k: int = st.slider("Hadiths to retrieve", min_value=3, max_value=10, value=5)

        st.divider()
        st.markdown("**ℹ️ About**")
        st.markdown(
            "Built by **Ubaid ur Rehman**  \n"
            "Aalim | AI Developer  \n"
            "[GitHub ↗](https://github.com/Ub207/hadith-ai-search)"
        )
        st.markdown(
            '<div class="credit">'
            "llama-3.3-70b-versatile · FAISS · sentence-transformers"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Chat history ──────────────────────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("hadiths"):
                st.markdown("**📖 Source Hadiths:**")
                for h in msg["hadiths"]:
                    render_hadith_expander(h)

    # ── Input ─────────────────────────────────────────────────────────────────
    pending = st.session_state.pop("_pending_query", None)
    user_input = st.chat_input(
        "Ask about any hadith in English or Urdu… مثلاً: صبر کے بارے میں حدیث بتائیں"
    )
    query = pending or user_input

    if not query:
        return

    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Retrieve + generate
    with st.chat_message("assistant"):
        with st.spinner("Searching hadith database…"):
            results = search_hadiths(query, model, index, metadata, k=k)
            context = build_context(results)

        with st.spinner("Generating scholarly answer…"):
            answer = generate_answer(client, query, context)

        st.markdown(answer)
        st.markdown("---")
        st.markdown("**📖 Source Hadiths:**")
        for h in results:
            render_hadith_expander(h)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "hadiths": results}
    )


if __name__ == "__main__":
    main()
