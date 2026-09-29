"""Download pretrained GloVe vectors and extract the file we need.

The vectors come from the Hugging Face mirror of the Stanford release, because
``hf_hub_download`` resumes interrupted transfers - the original Stanford link is
a plain HTTP download that starts from zero every time it drops.

Only ``glove.6B.100d.txt`` is extracted; the other three files in the archive
(50d/200d/300d) are not used by this project.

Usage
-----
    python -m src.download_glove
"""

from __future__ import annotations

import zipfile
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
GLOVE_DIR: Path = PROJECT_ROOT / "data" / "glove"

HF_REPO: str = "stanfordnlp/glove"
ZIP_NAME: str = "glove.6B.zip"
WANTED: str = "glove.6B.100d.txt"
EMBED_DIM: int = 100


def download_glove(force: bool = False) -> Path:
    """Return the path to ``glove.6B.100d.txt``, downloading it if needed."""
    target = GLOVE_DIR / WANTED
    if target.exists() and not force:
        size_mb = target.stat().st_size / 1024**2
        print(f"GloVe already present: {target} ({size_mb:.0f} MB)")
        return target

    from huggingface_hub import hf_hub_download

    GLOVE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ZIP_NAME} from {HF_REPO} (~822 MB, resumable)...")
    archive = hf_hub_download(repo_id=HF_REPO, filename=ZIP_NAME, repo_type="model")

    print(f"Extracting {WANTED}...")
    with zipfile.ZipFile(archive) as zf:
        with zf.open(WANTED) as src, open(target, "wb") as dst:
            while chunk := src.read(1024 * 1024):
                dst.write(chunk)

    size_mb = target.stat().st_size / 1024**2
    print(f"Ready: {target} ({size_mb:.0f} MB)")
    return target


if __name__ == "__main__":
    download_glove()
