#!/usr/bin/env python3
"""Download PPE Human Preference V1 (not stored in git).

The raw Chatbot Arena prompts contain a string that GitHub secret scanning flags as an API key,
so the file cannot be committed. This script fetches the exact official file and checks its
SHA-256 so every run uses identical data.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = ("https://huggingface.co/datasets/lmarena-ai/PPE-Human-Preference-V1/resolve/main/"
       "data/test-00000-of-00001.parquet")
SHA256 = "9b0207046265a83edc2326c512dc82c732ad873e51e9441e1c9820ccd64fbdfa"
TARGET = Path(__file__).resolve().parents[1] / "generating_dynamic_rubric" / "data" / "ppe" / "human_preference_v1_test.parquet"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, TARGET)
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if digest != SHA256:
        TARGET.unlink()
        sys.exit(f"SHA-256 mismatch ({digest}); the file was removed. Expected {SHA256}.")
    print(f"OK {TARGET}")


if __name__ == "__main__":
    main()
