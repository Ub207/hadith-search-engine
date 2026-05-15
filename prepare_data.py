#!/usr/bin/env python3
"""
Hadith AI Search Engine — Data Preparation
Downloads Sahih Bukhari & Sahih Muslim, builds FAISS semantic search index.
Run once: python prepare_data.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path("data")

COLLECTIONS = {
    "bukhari": {
        "name": "Sahih Bukhari",
        "english_url": "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-bukhari.json",
        "arabic_url": "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/ara-bukhari.json",
    },
    "muslim": {
        "name": "Sahih Muslim",
        "english_url": "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/eng-muslim.json",
        "arabic_url": "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/ara-muslim.json",
    },
}


def download_json(url: str, cache_path: Path) -> dict:
    if cache_path.exists():
        print(f"    [cached] {cache_path.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"    Downloading {url[:70]}...")
    resp = requests.get(url, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"    Saved: {cache_path.name}")
    return data


def parse_collection(eng_data: dict, ara_data: dict | None, collection_name: str) -> list[dict]:
    sections: dict[str, str] = eng_data.get("metadata", {}).get("sections", {})

    arabic_lookup: dict[int, str] = {}
    if ara_data:
        for h in ara_data.get("hadiths", []):
            n = h.get("hadithnumber")
            if n is not None:
                arabic_lookup[n] = (h.get("text") or "").strip()

    hadiths: list[dict] = []
    for h in eng_data.get("hadiths", []):
        num = h.get("hadithnumber")
        text = (h.get("text") or "").strip()
        if not text or num is None:
            continue

        ref = h.get("reference") or {}
        book_num = ref.get("book", "")
        book_name = sections.get(str(book_num), f"Book {book_num}") if book_num else "Unknown"

        search_text = f"{collection_name} {book_name} Hadith {num} {text}"

        hadiths.append({
            "collection": collection_name,
            "book_num": book_num,
            "book_name": book_name,
            "hadith_number": num,
            "text": text,
            "arabic_text": arabic_lookup.get(num, ""),
            "search_text": search_text,
        })

    return hadiths


def main() -> None:
    print("=" * 60)
    print("  Hadith AI Search Engine — Data Preparation")
    print("=" * 60)

    DATA_DIR.mkdir(exist_ok=True)
    all_hadiths: list[dict] = []

    for key, col in COLLECTIONS.items():
        print(f"\n[{col['name']}]")

        eng_data = download_json(col["english_url"], DATA_DIR / f"{key}_en.json")

        ara_data = None
        try:
            ara_data = download_json(col["arabic_url"], DATA_DIR / f"{key}_ar.json")
            print(f"    Arabic text loaded ✓")
        except Exception as e:
            print(f"    Arabic unavailable ({e})")

        parsed = parse_collection(eng_data, ara_data, col["name"])
        all_hadiths.extend(parsed)
        print(f"    Parsed: {len(parsed):,} hadiths")

    total = len(all_hadiths)
    print(f"\nTotal hadiths: {total:,}")

    # ── Embeddings ────────────────────────────────────────────────────────────
    print("\nLoading embedding model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings (this takes a few minutes on first run)...")
    texts = [h["search_text"] for h in all_hadiths]
    batch_size = 512
    all_embeddings: list[np.ndarray] = []

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)
        done = min(i + batch_size, total)
        print(f"  {done:,}/{total:,} ({done / total * 100:.0f}%)", end="\r", flush=True)

    print()
    embeddings = np.vstack(all_embeddings).astype("float32")

    # ── FAISS Index ───────────────────────────────────────────────────────────
    print("Building FAISS index (cosine similarity)...")
    import faiss

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("Saving index and metadata...")
    faiss.write_index(index, str(DATA_DIR / "hadith_index.faiss"))
    with open(DATA_DIR / "hadith_metadata.pkl", "wb") as f:
        pickle.dump(all_hadiths, f)

    print(f"\nDone! {index.ntotal:,} hadiths indexed.")
    print("  data/hadith_index.faiss")
    print("  data/hadith_metadata.pkl")
    print("\nNow run: streamlit run app.py")


if __name__ == "__main__":
    main()
