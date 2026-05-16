"""
One-time script to upload pre-built index files to HuggingFace Hub.
Run this ONCE from your local machine after prepare_data.py finishes.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # enter your HF token
    python upload_to_hf.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_ID   = "Ub207/hadith-search-engine-data"
DATA_DIR  = Path("data")
FILES     = ["hadith_index.faiss", "hadith_metadata.pkl"]

def main():
    api = HfApi()

    print(f"Creating dataset repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=False)

    for fname in FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"MISSING: {fpath} — run prepare_data.py first")
            continue
        size_mb = fpath.stat().st_size / 1_048_576
        print(f"Uploading {fname} ({size_mb:.1f} MB) ...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"  Done: https://huggingface.co/datasets/{REPO_ID}/blob/main/{fname}")

    print("\nAll files uploaded. Your Streamlit Cloud app will auto-download them on first run.")

if __name__ == "__main__":
    main()
