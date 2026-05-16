<div align="center">

# بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ

# 📚 Hadith AI Search Engine

### AI-Powered Semantic Search Across 35,975+ Authenticated Ahadith

[![Live Demo](https://img.shields.io/badge/🔍_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://ahadiths-search-engine.streamlit.app)
[![GitHub](https://img.shields.io/badge/⭐_Star_This_Repo-GitHub-181717?style=for-the-badge)](https://github.com/Ub207/hadith-search-engine)

**Built by [Ubaid ur Rehman](https://ubaid-portfolio-v2-zbbe.vercel.app) — Aalim (Islamic Scholar) & AI Developer**

*Combining 15+ years of Islamic scholarship with modern AI engineering*

</div>

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-0467DF?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Hadith Count](https://img.shields.io/badge/Ahadith-35%2C975%2B-gold?style=flat-square)
![Collections](https://img.shields.io/badge/Collections-7-blueviolet?style=flat-square)
![Languages](https://img.shields.io/badge/Languages-Arabic_%7C_English_%7C_Urdu-teal?style=flat-square)

</div>

---

## 📖 Overview

The **Hadith AI Search Engine** is a RAG-powered semantic search application that lets you search across all six major Hadith collections (Sihah Sitta) plus Muwatta Imam Malik — totaling 35,975+ authenticated ahadith — using plain English or Urdu questions. Unlike keyword search, this system understands the *meaning* of your query and retrieves the most contextually relevant hadith from across all collections simultaneously.

What makes this project truly unique is the combination required to build it: you need both **deep Islamic scholarship** (to correctly index, grade, and contextualize hadith) and **modern AI engineering** (RAG pipelines, vector embeddings, LLM orchestration). The AI answers are grounded exclusively in the indexed hadith database — it never fabricates or paraphrases without citation. Every result includes the original **Arabic matan (متن)**, full English and Urdu translations, and the authentic hadith grade (Sahih/Hasan/Daif).

---

## ✅ Features

- ✅ **Semantic search** across ALL Sihah Sitta (6 major hadith collections) + Muwatta Malik
- ✅ **AI-generated scholarly answers** with proper hadith citations
- ✅ **Arabic matan (متن عربی)** displayed for every hadith result
- ✅ **English and Urdu translations** side-by-side
- ✅ **Hadith grading** displayed: Sahih (صحیح) / Hasan (حسن) / Daif (ضعیف)
- ✅ **Filter by collection**, grade, and narrator
- ✅ **Browse Collections tab** — explore all 7 collections independently
- ✅ **Example questions** in sidebar for instant quick start
- ✅ **Responsive dark theme UI** with Blue & Gold design
- ✅ **Works with English and Urdu queries** — no Arabic input required
- ✅ **35,975+ ahadith indexed** with FAISS cosine similarity search
- ✅ **RAG (Retrieval-Augmented Generation)** — AI only cites from the actual indexed database, never fabricates

---

## 📚 Collections Indexed

| # | Collection | Arabic Name | Hadith Count | Arabic | English | Urdu |
|---|---|---|---|:---:|:---:|:---:|
| 1 | Sahih al-Bukhari | صحیح البخاری | 7,580 | ✅ | ✅ | ✅ |
| 2 | Sahih Muslim | صحیح مسلم | 7,360 | ✅ | ✅ | ✅ |
| 3 | Sunan Abu Dawud | سنن ابو داؤد | 5,272 | ✅ | ✅ | ✅ |
| 4 | Jami at-Tirmidhi | جامع ترمذی | 3,926 | ✅ | ✅ | ✅ |
| 5 | Sunan an-Nasa'i | سنن نسائی | 5,679 | ✅ | ✅ | ✅ |
| 6 | Sunan Ibn Majah | سنن ابن ماجہ | 4,340 | ✅ | ✅ | ✅ |
| 7 | Muwatta Imam Malik | موطا امام مالک | 1,818 | ❌ | ✅ | ✅ |
| | **Total** | | **35,975** | | | |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | [Streamlit](https://streamlit.io) |
| AI / LLM | [Groq API](https://console.groq.com) — Llama 3.3 70B Versatile |
| Embeddings | [Sentence Transformers](https://www.sbert.net) — `all-MiniLM-L6-v2` |
| Vector DB | [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search) |
| Data Source | [fawazahmed0/hadith-api](https://github.com/fawazahmed0/hadith-api) via jsDelivr CDN |
| Deployment | [Streamlit Cloud](https://streamlit.io/cloud) + [HuggingFace Hub](https://huggingface.co) |

---

## 🏗️ Architecture

```
User Query (English / Urdu)
          ↓
Sentence Transformer (all-MiniLM-L6-v2)
          ↓
      Query Embedding
          ↓
  FAISS Cosine Similarity Search
  (over 35,975 indexed hadith)
          ↓
    Top-K Relevant Hadith
          ↓
 Context Assembly
 (Arabic متن + English + Urdu + Grade + Chain)
          ↓
   Groq API (Llama 3.3 70B Versatile)
          ↓
Scholarly AI Answer with Arabic Matan + Citations
```

The system uses **Retrieval-Augmented Generation (RAG)**: the LLM never relies on its training data for hadith content. It *only* synthesises answers from the hadith retrieved by FAISS, ensuring authenticity and traceability of every citation.

---

## ⚙️ How It Works

1. **Data Collection** — `prepare_data.py` downloads all 7 hadith collections from the `fawazahmed0/hadith-api` CDN in three languages (Arabic, English, Urdu) and caches them locally.

2. **Embedding Generation** — Every hadith's English text is encoded into a dense vector using `sentence-transformers/all-MiniLM-L6-v2`, producing 35,975+ embeddings.

3. **FAISS Indexing** — All embeddings are added to a FAISS `IndexFlatIP` (inner product / cosine similarity) index and saved to disk as `hadith_index.faiss` alongside `hadith_metadata.pkl`.

4. **Query Processing** — A user's question is embedded with the same model, then FAISS retrieves the top-K most semantically similar hadith from across all collections in milliseconds.

5. **AI Answer Generation** — The retrieved hadith (with Arabic matan, translations, grades, and book references) are assembled into a structured context prompt. Groq's Llama 3.3 70B then generates a scholarly, well-cited answer grounded entirely in those hadith — no hallucination, no fabrication.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Ub207/hadith-search-engine.git
cd hadith-search-engine

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Prepare data — downloads all 7 hadith collections and builds the FAISS index
# (first run: ~5–10 min depending on internet speed; subsequent runs use cache)
python prepare_data.py

# Launch the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

**Getting a Groq API key (free):**
1. Visit [https://console.groq.com](https://console.groq.com)
2. Sign up or log in
3. Go to **API Keys** → **Create API Key**
4. Copy the key into your `.env` file

Groq provides free, ultra-fast inference for Llama 3.3 70B — no credit card required for standard usage.

---

## 📁 Project Structure

```
hadith-search-engine/
├── app.py                    # Main Streamlit application
├── prepare_data.py           # Data download + FAISS index builder
├── upload_to_hf.py           # Uploads built index to HuggingFace Hub
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── .gitignore
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration
└── data/                     # Generated data (not tracked in git)
    ├── hadith_index.faiss    # FAISS vector index (35,975 hadith)
    ├── hadith_metadata.pkl   # Pickled list of hadith dicts
    ├── hadith_combined.json  # All hadith in one JSON file
    ├── collections_meta.json # Per-collection stats and metadata
    ├── bukhari_en.json       # Cached collection downloads
    ├── bukhari_ar.json
    ├── bukhari_ur.json
    └── ...                   # (one _en / _ar / _ur per collection)
```

> **Note:** The `data/` directory is excluded from git (it's large). On Streamlit Cloud, the pre-built index is automatically downloaded from HuggingFace Hub at startup.

---

## 💡 Why This Project Is Unique

Building an Islamic AI tool is not a standard software engineering problem — it requires a rare combination of expertise that almost no developer possesses:

- **Reading Arabic fluently** — to verify hadith matan, not just transliterate
- **Understanding hadith sciences** — knowing the difference between Sahih, Hasan, Daif, and Mawdu; understanding isnad (narrator chains) and matn criticism
- **Collection-level knowledge** — knowing which topics are covered in which kitab, which ahadith overlap between Bukhari and Muslim, which are unique to Tirmidhi or Ibn Majah
- **Fiqh context** — understanding how different madhabs use different hadith, and why grade matters for legal rulings

With **15+ years of formal Islamic education** (Dars-e-Nizami, Quran memorisation, Saba Qiraat with Ijazah) combined with professional AI development training, this project bridges a gap that neither pure engineers nor pure scholars could fill alone.

The AI is intentionally constrained: it **only** cites from the 35,975 hadith in the indexed database. It will not guess, paraphrase from memory, or invent narrations. If a relevant hadith exists, it will be found and cited correctly — with the Arabic text to verify against.

---

## 👤 About the Developer

```
Ubaid ur Rehman
────────────────────────────────────────────────
🎓  Aalim (Islamic Scholar) — 15+ years teaching Quran & Hadith
📖  Qari — Saba Qiraat Specialist with Ijazah
🤖  AI Developer — GIAIC Generative AI Program
🌐  Portfolio     : https://ubaid-portfolio-v2-zbbe.vercel.app
🕌  Quran Academy : https://quranconnectacademy.netlify.app
💼  GitHub        : https://github.com/Ub207
📧  Contact       : usmanubaidurrehman@gmail.com
```

---

## 🔗 Related Projects

| Project | Description |
|---|---|
| [Islamic Fiqh Q&A Assistant](https://github.com/Ub207/fiqh-qa-bot) | AI chatbot for Hanafi fiqh questions |
| [Arabic Handwriting Recognition](https://github.com/Ub207/arabic-handwriting-checker) | AI recognition of handwritten Arabic letters |

---

## 🙏 Data Credits

| Resource | Source |
|---|---|
| Hadith data (all collections, all languages) | [fawazahmed0/hadith-api](https://github.com/fawazahmed0/hadith-api) |
| Embedding model | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| LLM inference | [Groq](https://console.groq.com) — Llama 3.3 70B Versatile |

All hadith data is used for educational and da'wah purposes. The `fawazahmed0/hadith-api` project deserves enormous credit for making authenticated hadith data freely accessible in machine-readable format.

---

## 📜 License

This project is released under the **MIT License** — free to use, modify, and distribute for personal, educational, and non-commercial purposes. If you build on this work, a credit or star would be appreciated. 🤲

---

<div align="center">

## ⭐ Support This Project

**If this tool benefits your Hadith studies, please consider:**

⭐ **Starring** this repository so others can discover it

🤲 **Sharing** with students, scholars, and anyone who studies Hadith

🐛 **Reporting bugs** by opening a [GitHub Issue](https://github.com/Ub207/hadith-search-engine/issues)

💡 **Suggesting features** by opening a [GitHub Discussion](https://github.com/Ub207/hadith-search-engine/discussions)

---

*"Convey from me, even if it is one verse."*
*— Sahih al-Bukhari 3461*

**جَزَاكُمُ اللَّهُ خَيْرًا**

</div>
