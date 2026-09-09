#!/usr/bin/env python3
"""
download_tng_snapshot.py

Download all HDF5 chunks of a given snapshot from the IllustrisTNG API,
verifying each chunk against the official sha256 checksums file, and
re-downloading any chunk that is missing or corrupted. Checking and
downloading chunks happens in parallel across a configurable thread pool.

Usage:
    python download_tng_snapshot.py --sim TNG50-1 --snap 97 \
        --api-key YOUR_API_KEY --outdir ./TNG50-1/snapdir_097 --workers 6

    # Optionally set the key via environment variable instead of --api-key:
    export TNG_API_KEY=YOUR_API_KEY
    python download_tng_snapshot.py --sim TNG100-2 --snap 97 --outdir ./data

Notes:
- Get your API key from your profile page at https://www.tng-project.org/users/profile/
  Never commit real API keys to source control.
- The IllustrisTNG file-download endpoint is:
      https://www.tng-project.org/api/<sim>/files/snapshot-<snap>.<chunk>.hdf5
  The server responds with a Content-Disposition header giving the real
  filename (e.g. snap_097.0.hdf5), which this script uses when saving.
- The checksums endpoint is:
      https://www.tng-project.org/api/<sim>/checksums/snapshot-<snap>/
  It returns lines of "<sha256>  <filename>" (sha256sum-style) for every
  file belonging to that snapshot. This script parses it to discover how
  many chunks exist and what their expected hashes are.
"""

import argparse
import hashlib
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

PRINT_LOCK = threading.Lock()


def log(msg: str) -> None:
    """Thread-safe print (avoids interleaved output from parallel workers)."""
    with PRINT_LOCK:
        print(msg, flush=True)

BASE_URL = "https://www.tng-project.org/api"


def get_checksums(sim: str, snap: int, api_key: str, session: requests.Session) -> dict:
    """Download and parse the checksums.txt for this snapshot.

    Returns a dict: {filename: sha256_hex} restricted to snapshot hdf5
    chunk files (snap_<snap>.<chunk>.hdf5), keyed by chunk index (int).
    """
    url = f"{BASE_URL}/{sim}/checksums/snapshot-{snap}/"
    headers = {"API-Key": api_key}

    resp = session.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    text = resp.text

    # Pattern for the actual snapshot data chunks, e.g. "snap_097.12.hdf5"
    chunk_re = re.compile(r"snap_(\d+)\.(\d+)\.hdf5$")

    chunks = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        sha256_hex, filename = parts[0], parts[-1]
        filename = filename.lstrip("*")  # some sha256sum formats prefix binary marker with *
        m = chunk_re.search(filename)
        if not m:
            continue  # skip offset files / other non-chunk entries
        chunk_idx = int(m.group(2))
        chunks[chunk_idx] = {"filename": os.path.basename(filename), "sha256": sha256_hex.lower()}

    if not chunks:
        raise RuntimeError(
            f"No snapshot chunk entries found in checksums for {sim} snapshot-{snap}. "
            "Check the simulation name / snapshot number / API key."
        )

    return chunks


def sha256_of_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def file_is_valid(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        return sha256_of_file(path) == expected_sha256.lower()
    except OSError:
        return False


def download_chunk(sim: str, snap: int, chunk_idx: int, dest_path: Path,
                    api_key: str, session: requests.Session, timeout: int = 300) -> float:
    """Stream-download one snapshot chunk to dest_path (via a .part temp file).

    Each worker uses its own .part temp file (named after dest_path + chunk
    index), so concurrent workers never collide on the same temp path.
    Returns the number of MB written.
    """
    url = f"{BASE_URL}/{sim}/files/snapshot-{snap}.{chunk_idx}.hdf5"
    headers = {"API-Key": api_key}

    tmp_path = dest_path.with_suffix(dest_path.suffix + f".part{chunk_idx}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with session.get(url, headers=headers, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            for block in resp.iter_content(chunk_size=1024 * 1024):
                if block:
                    f.write(block)
                    written += len(block)
    tmp_path.replace(dest_path)
    return written / 1e6


def process_chunk(sim: str, snap: int, chunk_idx: int, n_chunks: int, info: dict,
                   outdir: Path, api_key: str, max_retries: int,
                   session: requests.Session, check_only: bool = False) -> tuple:
    """Check + (re)download a single chunk, with retries. Runs in a worker thread.

    If check_only is True, only verifies existence/checksum and never
    downloads anything.

    Returns (chunk_idx, status) where status is one of:
    "ok", "downloaded", "failed" (normal mode), or
    "ok", "missing", "mismatch" (check_only mode).
    """
    dest_path = outdir / info["filename"]
    expected = info["sha256"]
    tag = f"[{chunk_idx}/{n_chunks - 1}] {info['filename']}"

    if file_is_valid(dest_path, expected):
        log(f"{tag}: OK (checksum matches).")
        return chunk_idx, "ok"

    if check_only:
        if dest_path.exists():
            log(f"{tag}: checksum MISMATCH.")
            return chunk_idx, "mismatch"
        else:
            log(f"{tag}: MISSING.")
            return chunk_idx, "missing"

    if dest_path.exists():
        log(f"{tag}: checksum MISMATCH or unreadable, re-downloading.")
    else:
        log(f"{tag}: missing, downloading.")

    for attempt in range(1, max_retries + 1):
        try:
            mb = download_chunk(sim, snap, chunk_idx, dest_path, api_key, session)
        except (requests.RequestException, OSError) as e:
            log(f"{tag}: attempt {attempt}/{max_retries} failed to download: {e}")
            time.sleep(min(2 ** attempt, 30))
            continue

        if file_is_valid(dest_path, expected):
            log(f"{tag}: downloaded {mb:.1f} MB, checksum verified.")
            return chunk_idx, "downloaded"
        else:
            log(f"{tag}: attempt {attempt}/{max_retries}: checksum still does not match, retrying.")
            time.sleep(min(2 ** attempt, 30))

    log(f"{tag}: GIVING UP after {max_retries} attempts.")
    return chunk_idx, "failed"


def main():
    parser = argparse.ArgumentParser(description="Download & verify IllustrisTNG snapshot chunks.")
    parser.add_argument("--sim", required=True, help="Simulation name, e.g. TNG50-1, TNG100-2")
    parser.add_argument("--snap", required=True, type=int, help="Snapshot number, e.g. 97")
    parser.add_argument("--api-key", default=os.environ.get("TNG_API_KEY"),
                         help="TNG API key (or set TNG_API_KEY env var)")
    parser.add_argument("--outdir", default=None,
                         help="Output directory (default: ./<sim>/snapdir_<snap>)")
    parser.add_argument("--max-retries", type=int, default=5,
                         help="Max download attempts per chunk before giving up (default: 5)")
    parser.add_argument("--chunks", default=None,
                         help="Optional comma-separated list / range of chunk indices to "
                              "restrict to, e.g. '0-9,15,20-25'. Default: all chunks found.")
    parser.add_argument("--workers", type=int, default=4,
                         help="Number of chunks to check/download in parallel (default: 4). "
                              "Be considerate of the TNG server; 4-8 is usually plenty.")
    parser.add_argument("--check", action="store_true",
                         help="Only check whether all chunk files exist and have the correct "
                              "sha256 checksum. Does not download anything. Exits non-zero if "
                              "any chunk is missing or mismatched.")
    args = parser.parse_args()

    if not args.api_key:
        sys.exit("ERROR: no API key provided. Use --api-key or set TNG_API_KEY.")

    snap_str = f"{args.snap:03d}"
    outdir = Path(args.outdir) if args.outdir else Path(args.sim) / f"snapdir_{snap_str}"
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    print(f"Fetching checksums for {args.sim} snapshot-{args.snap} ...")
    chunks = get_checksums(args.sim, args.snap, args.api_key, session)
    n_chunks = max(chunks.keys()) + 1
    print(f"Found {len(chunks)} chunk entries (indices 0..{n_chunks - 1}).")

    wanted = set(chunks.keys())
    if args.chunks:
        wanted = set()
        for part in args.chunks.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                wanted.update(range(int(lo), int(hi) + 1))
            else:
                wanted.add(int(part))
        wanted &= set(chunks.keys())

    ok, redownloaded, failed = [], [], []
    missing, mismatch = [], []

    # One shared requests.Session (connection pool is thread-safe enough for
    # this use case), sized so the pool doesn't bottleneck the worker count.
    session.mount(
        "https://",
        requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers),
    )

    if args.check:
        log(f"Checking {len(wanted)} chunk(s) with {args.workers} parallel worker(s) "
            f"(check-only, no downloads)...")
    else:
        log(f"Checking/downloading {len(wanted)} chunk(s) with {args.workers} parallel worker(s)...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_chunk, args.sim, args.snap, chunk_idx, n_chunks,
                             chunks[chunk_idx], outdir, args.api_key, args.max_retries,
                             session, args.check): chunk_idx
            for chunk_idx in sorted(wanted)
        }
        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                _, status = future.result()
            except Exception as e:  # noqa: BLE001 - surface unexpected worker errors
                log(f"[{chunk_idx}] unexpected error: {e}")
                status = "failed"

            if status == "ok":
                ok.append(chunk_idx)
            elif status == "downloaded":
                redownloaded.append(chunk_idx)
            elif status == "missing":
                missing.append(chunk_idx)
            elif status == "mismatch":
                mismatch.append(chunk_idx)
            else:
                failed.append(chunk_idx)

    print("\n===== Summary =====")
    if args.check:
        print(f"OK:        {len(ok)}")
        print(f"Missing:   {len(missing)}  {sorted(missing) if missing else ''}")
        print(f"Mismatch:  {len(mismatch)}  {sorted(mismatch) if mismatch else ''}")
        if missing or mismatch:
            sys.exit(1)
        return

    print(f"Already OK:     {len(ok)}")
    print(f"Re-downloaded:  {len(redownloaded)}")
    print(f"Failed:         {len(failed)}  {failed if failed else ''}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()