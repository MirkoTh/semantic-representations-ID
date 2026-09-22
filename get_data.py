#!/usr/bin/env python3
"""Download external data from OSF into data/. Skips files already present.

Each item is pinned to a known SHA-256 (see HASHES). To fill HASHES the first
time, run `python get_data.py --print-hashes`, paste the printed values in, and
commit this file. After that, every run verifies downloads against those hashes.

Usage:
    python get_data.py                 # download + verify
    python get_data.py --print-hashes  # download all, print hashes to paste into HASHES

Requires: requests, tqdm
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

DATA = Path("data")
URL = "https://osf.io/{guid}/download"

# guid -> filename in data/ for the flat files.
FILES = {
    "h4m8a": "labels.mat",
    "qe8sj": "words.mat",
    "2y463": "unique_id.txt",
    "h2smy": "triplets_large_final_correctednc_correctedorder.csv",
}
# The images folder is fetched as a zip, unpacked, then verified by a single
# combined hash of its *contents* (not the zip wrapper, whose bytes aren't
# guaranteed stable across downloads). The zip is password-protected; the
# password is published with the source dataset.
IMAGES_GUID = "rdxy2"
IMAGES_ZIP_PASSWORD = b"things4all"

# Pinned SHA-256 for each item. Fill via `--print-hashes`, then commit.
# "images/" is the combined hash of the unpacked image files.
HASHES = {
    "labels.mat": "18f997190e922b9c131b72f97efc21045325bc08d4bb0917f572ea343032bf5c",
    "words.mat": "6b18fabf86c5d7e69d5f34c65dfa9625a3c08b4e078a5a0f83870f7e90a031ad",
    "unique_id.txt": "e2e59d02197e89776034938e7a73940a0823cacac4b6084b56b1d8959816ff76",
    "triplets_large_final_correctednc_correctedorder.csv": "68f2bbc330b17537881f81b1aa806fc2179b7accecc3a44d54fb25ae3de7f3e5",
    "images/": "ad73f732ca3bf5b1da15594c7af858f02e26449458e4a50ee6e7ecd9abfed3b9",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_dir(root):
    """Combined SHA-256 of a folder's contents. Files are processed in sorted
    relative-path order; each file's path and bytes are folded into one digest,
    so a missing, extra, renamed, or altered file changes the result. Immune to
    zip-wrapper differences since it hashes the files themselves."""
    h = hashlib.sha256()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        h.update(str(p.relative_to(root)).encode())  # bind the name
        h.update(b"\0")
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def download(guid, dest, retries=10):
    """Download to dest, resuming across interruptions.

    Streams into a .part file. If a partial .part already exists, continues from
    where it left off via an HTTP Range request instead of restarting — important
    for the large (~5 GB) images zip over unstable connections. Retries on broken
    connections, verifies the final size against the server's Content-Length, and
    only then renames into place, so a truncated download never looks complete.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = URL.format(guid=guid)

    # Full expected size (follow redirects; OSF redirects to storage).
    try:
        head = requests.head(url, allow_redirects=True, timeout=30)
        total = int(head.headers.get("Content-Length", 0)) or None
    except requests.RequestException:
        total = None

    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if total and have >= total:
            break  # already fully downloaded, just needs finalizing
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, timeout=60, headers=headers,
                              allow_redirects=True) as r:
                # 206 = server honored resume; 200 = full send (restart).
                if have and r.status_code == 200:
                    have = 0  # server ignored Range; start over
                r.raise_for_status()
                mode = "ab" if have else "wb"
                with open(part, mode) as f, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True,
                    desc=dest.name, leave=False
                ) as bar:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        bar.update(len(chunk))
            break  # finished the stream without error
        except (requests.RequestException, OSError) as e:
            if attempt == retries:
                raise RuntimeError(
                    f"{dest.name}: download failed after {retries} attempts "
                    f"({e}). Re-run to resume from {part.stat().st_size if part.exists() else 0} bytes.")
            print(f"  … {dest.name}: connection dropped, resuming "
                  f"(attempt {attempt}/{retries})")

    # Final size check before accepting the file.
    final = part.stat().st_size
    if total and final != total:
        raise RuntimeError(
            f"{dest.name}: size mismatch (got {final}, expected {total}). "
            f"Re-run to resume.")
    os.replace(part, dest)


def find_7zip():
    """Locate a 7-Zip executable, or None. Checks PATH names and common Windows
    install locations."""
    for name in ("7z", "7za", "7zz", "7z.exe", "7za.exe"):
        found = shutil.which(name)
        if found:
            return found
    for guess in (
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if os.path.isfile(guess):
            return guess
    return None


def _flatten_single_wrapper(dest):
    """If dest contains exactly one top-level directory (OSF nests contents
    under e.g. 'object_images/'), lift its children up into dest."""
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        wrapper = entries[0]
        for child in wrapper.iterdir():
            shutil.move(str(child), str(dest / child.name))
        wrapper.rmdir()


def extract_zip(zip_path, dest, password=None):
    """Extract a (possibly password-protected) zip into dest.

    Uses 7-Zip if available (native, fast — important here because Python's
    stdlib ZipCrypto decryption is pure-Python and extremely slow for the
    ~25k image files). Falls back to stdlib zipfile otherwise. Either way the
    files on disk end up identical, so the pinned content hash matches
    regardless of which extractor ran.

    Also flattens a single top-level wrapper directory so files land directly
    in dest.
    """
    dest.mkdir(parents=True, exist_ok=True)
    seven = find_7zip()

    if seven:
        cmd = [seven, "x", str(zip_path), f"-o{dest}", "-y", "-bd"]
        if password is not None:
            pw = password.decode() if isinstance(password, bytes) else password
            cmd.append(f"-p{pw}")
        print(f"  unzipping with 7-Zip ({seven})…")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"7-Zip extraction failed (exit {result.returncode}):\n"
                f"{result.stderr or result.stdout}")
    else:
        print("  7-Zip not found; using Python (slow for encrypted zips)…")
        with zipfile.ZipFile(zip_path) as zf:
            if password is not None:
                zf.setpassword(password)
            for info in tqdm(zf.infolist(), desc="unzip", leave=False):
                zf.extract(info, dest)

    _flatten_single_wrapper(dest)


def verify(name, path):
    got = sha256_dir(path) if path.is_dir() else sha256(path)
    want = HASHES.get(name, "")
    if not want:
        sys.exit(f"  x {name}: no pinned hash set. Run with --print-hashes first.")
    if got != want:
        sys.exit(f"  x {name}: HASH MISMATCH.\n"
                 f"      expected {want}\n      got      {got}\n"
                 f"    Remove {path} and re-run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-hashes", action="store_true",
                    help="download everything and print SHA-256 values for HASHES")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)

    if args.print_hashes:
        print("Paste these into HASHES:\n")
        for guid, name in FILES.items():
            dest = DATA / name
            download(guid, dest)
            print(f'    "{name}": "{sha256(dest)}",')
        # Images: download zip, unpack, hash the unpacked contents.
        images = DATA / "images"
        zip_path = DATA / "images.zip"
        download(IMAGES_GUID, zip_path)
        extract_zip(zip_path, images, IMAGES_ZIP_PASSWORD)
        zip_path.unlink()
        print(f'    "images/": "{sha256_dir(images)}",')
        print("\nDone. Paste the values above into HASHES and commit.")
        return

    # Flat files.
    for guid, name in FILES.items():
        dest = DATA / name
        if dest.exists():
            print(f"  skip {name} (exists)")
        else:
            download(guid, dest)
            verify(name, dest)
            print(f"  got  {name} (verified)")

    # Images: download zip, unpack, verify the unpacked contents, discard zip.
    images = DATA / "images"
    if images.exists():
        print("  skip images/ (exists)")
    else:
        zip_path = DATA / "images.zip"
        download(IMAGES_GUID, zip_path)
        extract_zip(zip_path, images, IMAGES_ZIP_PASSWORD)
        zip_path.unlink()
        verify("images/", images)
        n = sum(1 for _ in images.rglob("*") if _.is_file())
        print(f"  got  images/ ({n} files, verified)")

    print("Done.")


if __name__ == "__main__":
    main()