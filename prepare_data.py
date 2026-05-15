#!/usr/bin/env python3
"""
Hadith AI Search Engine - Data Preparation (Full Sihah Sitta)
Downloads all 6 major Hadith collections in Arabic + English + Urdu,
then builds a FAISS semantic search index.

Run once:  python prepare_data.py
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path("data")

CDN_URL = "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/{edition}.json"
RAW_URL = "https://raw.githubusercontent.com/fawazahmed0/hadith-api/1/editions/{edition}.json"

# ── Collection definitions ────────────────────────────────────────────────────
SIHAH_SITTA = [
    {
        "key": "bukhari",
        "name": "Sahih al-Bukhari",
        "arabic": "ara-bukhari",
        "english": "eng-bukhari",
        "urdu": "urd-bukhari",
    },
    {
        "key": "muslim",
        "name": "Sahih Muslim",
        "arabic": "ara-muslim",
        "english": "eng-muslim",
        "urdu": "urd-muslim",
    },
    {
        "key": "abudawud",
        "name": "Sunan Abu Dawud",
        "arabic": "ara-abudawud",
        "english": "eng-abudawud",
        "urdu": "urd-abudawud",
    },
    {
        "key": "tirmidhi",
        "name": "Jami at-Tirmidhi",
        "arabic": "ara-tirmidhi",
        "english": "eng-tirmidhi",
        "urdu": "urd-tirmidhi",
    },
    {
        "key": "nasai",
        "name": "Sunan an-Nasa'i",
        "arabic": "ara-nasai",
        "english": "eng-nasai",
        "urdu": "urd-nasai",
    },
    {
        "key": "ibnmajah",
        "name": "Sunan Ibn Majah",
        "arabic": "ara-ibnmajah",
        "english": "eng-ibnmajah",
        "urdu": "urd-ibnmajah",
    },
]

ADDITIONAL_COLLECTIONS = [
    {
        "key": "malik",
        "name": "Muwatta Imam Malik",
        "arabic": None,
        "english": "eng-malik",
        "urdu": "urd-malik",
    },
]


def download_json(edition_key: str, cache_path: Path) -> dict | None:
    """Download edition JSON with CDN → GitHub raw fallback. Caches to disk."""
    if cache_path.exists():
        print(f"    [cached] {cache_path.name}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    for label, url_template in [("CDN", CDN_URL), ("GitHub", RAW_URL)]:
        url = url_template.format(edition=edition_key)
        print(f"    Trying {label}: {edition_key} ...")
        try:
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            cache_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            print(f"    Saved → {cache_path.name}")
            return data
        except Exception as exc:
            print(f"    {label} FAILED ({edition_key}): {exc}")

    print(f"    Both sources failed for {edition_key} — skipping")
    return None


def get_grade_summary(grades: list) -> str:
    if not grades:
        return ""
    grade_vals = [g.get("grade", "") for g in grades if g.get("grade")]
    for g in grade_vals:
        if "Sahih" in g or "صحيح" in g:
            return "Sahih"
    for g in grade_vals:
        if "Hasan" in g or "حسن" in g:
            return "Hasan"
    for g in grade_vals:
        if "Daif" in g or "Da'if" in g or "ضعيف" in g:
            return "Daif"
    return grade_vals[0] if grade_vals else ""


def parse_collection(
    eng_data: dict,
    ara_data: dict | None,
    urd_data: dict | None,
    collection_name: str,
    collection_key: str,
) -> list[dict]:
    sections: dict[str, str] = eng_data.get("metadata", {}).get("sections", {})

    arabic_lookup: dict[int, str] = {}
    if ara_data:
        for h in ara_data.get("hadiths", []):
            n = h.get("hadithnumber")
            if n is not None:
                arabic_lookup[n] = (h.get("text") or "").strip()

    urdu_lookup: dict[int, str] = {}
    if urd_data:
        for h in urd_data.get("hadiths", []):
            n = h.get("hadithnumber")
            if n is not None:
                urdu_lookup[n] = (h.get("text") or "").strip()

    hadiths: list[dict] = []
    for h in eng_data.get("hadiths", []):
        num = h.get("hadithnumber")
        eng_text = (h.get("text") or "").strip()
        if not eng_text or num is None:
            continue

        ref = h.get("reference") or {}
        book_num = ref.get("book", "")
        book_name = (
            sections.get(str(book_num), f"Book {book_num}") if book_num else "Unknown"
        )

        grades = h.get("grades", [])
        grade_summary = get_grade_summary(grades)
        arabic_text = arabic_lookup.get(num, "")
        urdu_text = urdu_lookup.get(num, "")

        narrator_chain = ""
        if eng_text.startswith("Narrated") and ":" in eng_text[:200]:
            narrator_chain = eng_text.split(":", 1)[0].replace("Narrated ", "").strip()

        search_parts = [
            f"{collection_name} {book_name} Hadith {num}",
            eng_text,
        ]
        if urdu_text:
            search_parts.append(urdu_text)
        search_text = " | ".join(search_parts)

        hadiths.append(
            {
                "collection": collection_name,
                "collection_key": collection_key,
                "hadith_number": num,
                "book_number": book_num,
                "book_name": book_name,
                "arabic": arabic_text,
                "english": eng_text,
                "urdu": urdu_text,
                "grades": grades,
                "grade_summary": grade_summary,
                "narrator_chain": narrator_chain,
                "search_text": search_text,
            }
        )

    return hadiths


def main() -> None:
    print("=" * 65)
    print("  Hadith AI Search Engine - Full Sihah Sitta Data Preparation")
    print("=" * 65)

    DATA_DIR.mkdir(exist_ok=True)
    all_hadiths: list[dict] = []
    collections_meta: dict = {}

    all_collections = SIHAH_SITTA + ADDITIONAL_COLLECTIONS

    for col in all_collections:
        print(f"\n{'-' * 50}")
        print(f"  [{col['name']}]")

        eng_data = download_json(col["english"], DATA_DIR / f"{col['key']}_en.json")
        if eng_data is None:
            print(f"  SKIPPING {col['name']} — English data unavailable")
            continue

        ara_data = None
        if col.get("arabic"):
            ara_data = download_json(col["arabic"], DATA_DIR / f"{col['key']}_ar.json")
            print("    Arabic: OK" if ara_data else "    Arabic: unavailable (will skip)")

        urd_data = None
        if col.get("urdu"):
            urd_data = download_json(col["urdu"], DATA_DIR / f"{col['key']}_ur.json")
            print("    Urdu:   OK" if urd_data else "    Urdu:   unavailable (English only)")

        parsed = parse_collection(
            eng_data, ara_data, urd_data, col["name"], col["key"]
        )
        all_hadiths.extend(parsed)

        sections = eng_data.get("metadata", {}).get("sections", {})
        collections_meta[col["key"]] = {
            "name": col["name"],
            "count": len(parsed),
            "sections": sections,
            "has_arabic": ara_data is not None,
            "has_urdu": urd_data is not None,
        }

        ar_flag = "Arabic:Yes" if ara_data else "Arabic:No"
        ur_flag = "Urdu:Yes" if urd_data else "Urdu:No"
        print(f"    Parsed: {len(parsed):,} hadiths  ({ar_flag}  {ur_flag})")

    total = len(all_hadiths)
    print(f"\n{'=' * 65}")
    print(f"  Total hadiths combined: {total:,}")

    if total == 0:
        print("  ERROR: No hadiths loaded. Aborting.")
        sys.exit(1)

    # Save combined JSON + metadata
    print("\nSaving combined JSON and collection metadata ...")
    combined_path = DATA_DIR / "hadith_combined.json"
    meta_path = DATA_DIR / "collections_meta.json"
    combined_path.write_text(
        json.dumps(all_hadiths, ensure_ascii=False), encoding="utf-8"
    )
    meta_path.write_text(
        json.dumps(collections_meta, ensure_ascii=False), encoding="utf-8"
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    print("\nLoading embedding model (all-MiniLM-L6-v2) ...")
    from sentence_transformers import SentenceTransformer

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generating embeddings for {total:,} hadiths (batch size 256) ...")
    texts = [h["search_text"] for h in all_hadiths]
    batch_size = 256
    all_embeddings: list[np.ndarray] = []

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        embs = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.append(embs)
        done = min(i + batch_size, total)
        print(f"  Encoded {done:,}/{total:,} ({done / total * 100:.0f}%)", end="\r", flush=True)

    print()
    embeddings = np.vstack(all_embeddings).astype("float32")

    # ── FAISS index ───────────────────────────────────────────────────────────
    print("Building FAISS cosine-similarity index ...")
    import faiss

    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # ── Save index + metadata pickle ──────────────────────────────────────────
    print("Saving FAISS index and metadata pickle ...")
    faiss.write_index(index, str(DATA_DIR / "hadith_index.faiss"))
    with open(DATA_DIR / "hadith_metadata.pkl", "wb") as f:
        pickle.dump(all_hadiths, f)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  Done! {index.ntotal:,} hadiths indexed.\n")
    print("  Collection breakdown:")
    for key, meta in collections_meta.items():
        ar = "Arabic:Yes" if meta["has_arabic"] else "Arabic:No"
        ur = "Urdu:Yes" if meta["has_urdu"] else "Urdu:No"
        print(f"    {meta['name']:<32} {meta['count']:>6,}  ({ar}  {ur})")

    print(f"\n  Files saved to {DATA_DIR}/")
    print("    hadith_index.faiss")
    print("    hadith_metadata.pkl")
    print("    hadith_combined.json")
    print("    collections_meta.json")
    print(f"\n  Now run:  streamlit run app.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
