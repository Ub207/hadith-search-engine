"""
One-time script — upload pre-built FAISS data to HuggingFace Hub so
Streamlit Cloud can download them on first launch.

Usage (run once, locally):
    pip install huggingface_hub
    huggingface-cli login          # paste a WRITE token from huggingface.co/settings/tokens
    python upload_to_hf.py
"""

import re
from pathlib import Path
from huggingface_hub import HfApi, create_repo

REPO_NAME = "hadith-search-engine-data"
DATA_DIR  = Path("data")
APP_FILE  = Path("app.py")

# Only the 3 files the app needs at runtime (hadith_combined.json is not needed)
FILES = [
    "hadith_index.faiss",
    "hadith_metadata.pkl",
    "collections_meta.json",
]


def patch_app_repo_id(repo_id: str) -> None:
    """Update HF_REPO_ID in app.py so the cloud download points to the right repo."""
    text = APP_FILE.read_text(encoding="utf-8")
    new_text = re.sub(
        r'^(HF_REPO_ID\s*=\s*)["\'].*?["\']',
        f'HF_REPO_ID = "{repo_id}"',
        text,
        flags=re.MULTILINE,
    )
    if new_text != text:
        APP_FILE.write_text(new_text, encoding="utf-8")
        print(f"   ✅ app.py → HF_REPO_ID updated to \"{repo_id}\"")
    else:
        print(f"   ℹ  app.py HF_REPO_ID already set (or pattern not found)")


def main():
    api = HfApi()

    # Auto-detect logged-in username
    try:
        username = api.whoami()["name"]
    except Exception:
        print("\n❌ Not logged in to HuggingFace.")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a token with  WRITE  role")
        print("   3. Run:  huggingface-cli login  and paste the token")
        return

    repo_id = f"{username}/{REPO_NAME}"
    print(f"\n✅ Logged in as: {username}")
    print(f"   Dataset repo : https://huggingface.co/datasets/{repo_id}\n")

    # Create public dataset repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    print("   Repo ready.\n")

    # Upload each file
    all_ok = True
    for fname in FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"❌ MISSING: {fpath}  — run prepare_data.py first")
            all_ok = False
            continue
        size_mb = fpath.stat().st_size / 1_048_576
        print(f"⬆  {fname}  ({size_mb:.0f} MB) — uploading, please wait…")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"   ✅ https://huggingface.co/datasets/{repo_id}/blob/main/{fname}\n")

    if not all_ok:
        print("\nSome files were missing. Fix them and re-run.")
        return

    # Patch app.py with the correct repo id
    print("Patching app.py …")
    patch_app_repo_id(repo_id)

    print("\n" + "=" * 60)
    print("Upload complete!")
    print(f"\nNext steps:")
    print(f"  1. git add app.py requirements.txt upload_to_hf.py")
    print(f"  2. git commit -m 'fix: HF upload complete, HF_REPO_ID updated'")
    print(f"  3. git push origin main")
    print(f"\nStreamlit Cloud will auto-redeploy and download the data")
    print(f"from  https://huggingface.co/datasets/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
