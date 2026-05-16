"""
One-time script to upload pre-built index files to HuggingFace Hub.
Run this ONCE from your local machine after prepare_data.py finishes.

Usage:
    pip install huggingface_hub
    huggingface-cli login          # paste a WRITE-access token when prompted
    python upload_to_hf.py
"""

from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_NAME = "hadith-search-engine-data"   # just the repo name, username is auto-detected
DATA_DIR  = Path("data")
FILES     = ["hadith_index.faiss", "hadith_metadata.pkl"]


def main():
    api = HfApi()

    # Auto-detect the logged-in HuggingFace username
    try:
        user_info = api.whoami()
        username = user_info["name"]
    except Exception:
        print("\n❌ Not logged in to HuggingFace.")
        print("   Run:  huggingface-cli login")
        print("   Then paste a token with WRITE permission from:")
        print("   https://huggingface.co/settings/tokens\n")
        return

    repo_id = f"{username}/{REPO_NAME}"
    print(f"\n✅ Logged in as: {username}")
    print(f"   Repo will be: https://huggingface.co/datasets/{repo_id}\n")

    # Create the dataset repo (public, free)
    print(f"Creating dataset repo: {repo_id}")
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    print("   Repo ready.\n")

    # Upload each file
    for fname in FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"❌ MISSING: {fpath} — run prepare_data.py first")
            continue
        size_mb = fpath.stat().st_size / 1_048_576
        print(f"⬆  Uploading {fname} ({size_mb:.0f} MB) — please wait...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"   ✅ https://huggingface.co/datasets/{repo_id}/blob/main/{fname}\n")

    print("=" * 60)
    print(f"All files uploaded!")
    print(f"\nNow update HF_REPO_ID in app.py to:")
    print(f'   HF_REPO_ID = "{repo_id}"')
    print(f"\nThen add this to Streamlit Cloud secrets:")
    print(f"   (no secret needed for a public repo)")
    print("=" * 60)


if __name__ == "__main__":
    main()
