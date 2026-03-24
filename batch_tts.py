#!/usr/bin/env python3
import argparse
import base64
import csv
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import requests


def detect_delimiter(sample: str) -> str:
    # Same preference order as your trainer: tab, semicolon, comma.
    if "\t" in sample:
        return "\t"
    if ";" in sample:
        return ";"
    return ","


def read_csv_any_delim(path: Path) -> Tuple[List[List[str]], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        pos = f.tell()
        sample = f.read(4096)
        f.seek(pos)
        delim = detect_delimiter(sample)
        reader = csv.reader(f, delimiter=delim)
        rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    return rows, delim


def write_csv(path: Path, rows: List[List[str]], delim: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=delim)
        for r in rows:
            w.writerow(r)


_slug_re = re.compile(r"[^a-zA-Z0-9]+", re.UNICODE)


def safe_slug(text: str, max_len: int = 48) -> str:
    t = text.strip().lower()
    t = _slug_re.sub("_", t).strip("_")
    if not t:
        t = "item"
    return t[:max_len]


def sha1_short(text: str, n: int = 10) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def pick_first(d: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def looks_like_base64(s: str) -> bool:
    if not isinstance(s, str):
        return False
    if len(s) < 40:
        return False
    # base64 charset (allow = padding)
    return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", s))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def call_tts(
    url: str,
    text: str,
    cfg_scale: float,
    voice_prompt: Optional[str],
    repetition_penalty: float,
    top_p: float,
    temperature: float,
    do_sample: bool,
    max_new_tokens: int,
    trim_start: float,
    trim_end: float,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "text": text,
        "cfg_scale": cfg_scale,
        "voice_prompt": voice_prompt,
        "top_p": top_p,
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "trim_start_s": trim_start,
        "trim_end_s": trim_end,
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    j = r.json()
    if not isinstance(j, dict):
        return {"_raw": j}
    return j


def materialize_audio(
    tts_json: Dict[str, Any],
    out_wav_path: Path,
    session: requests.Session,
    timeout_s: int,
) -> None:
    """
    Writes WAV to out_wav_path using one of:
      - returned local file path (copy)
      - returned base64 audio (decode)
      - returned audio URL (download)
    """
    # 1) local path returned by server
    local_path = pick_first(
        tts_json,
        ["wav_path", "audio_path", "path", "file", "filename", "outfile", "output_path"],
    )
    if isinstance(local_path, str) and local_path.strip():
        lp = Path(local_path)
        if lp.exists() and lp.is_file():
            ensure_dir(out_wav_path.parent)
            shutil.copyfile(lp, out_wav_path)
            return

    # 2) base64 audio returned
    b64 = pick_first(
        tts_json,
        ["audio_base64", "wav_base64", "audio_b64", "wav_b64", "base64", "audio"],
    )
    if isinstance(b64, str) and looks_like_base64(b64):
        data = base64.b64decode(b64)
        ensure_dir(out_wav_path.parent)
        out_wav_path.write_bytes(data)
        return

    # 3) audio URL returned
    audio_url = pick_first(tts_json, ["audio_url", "wav_url", "url", "download_url"])
    if isinstance(audio_url, str) and audio_url.strip().startswith(("http://", "https://")):
        resp = session.get(audio_url, timeout=timeout_s)
        resp.raise_for_status()
        ensure_dir(out_wav_path.parent)
        out_wav_path.write_bytes(resp.content)
        return

    raise RuntimeError(
        "Don't know how to get WAV from server response. "
        "Expected a local path key (wav_path/path/...), base64 key (audio_base64/...), or audio_url.\n"
        f"Response keys: {sorted(tts_json.keys())}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Batch-generate TTS WAVs for a CSV and write wav paths back into a new CSV column."
    )
    ap.add_argument("csv", help="Input CSV path (e.g. lesson1.csv)")
    ap.add_argument("--out-csv", default=None, help="Output CSV path (default: <input>_with_tts.csv)")
    ap.add_argument("--audio-dir", default=None, help="Directory for WAV files (default: <input>_audio next to CSV)")
    ap.add_argument("--col", type=int, default=1, help="0-based column index to speak (default: 1 = 2nd column)")
    ap.add_argument("--skip-header", action="store_true", help="If set, treats first row as header and does not synthesize it")
    ap.add_argument("--wav-col-name", default="tts_wav", help="Header name for the wav path column (default: tts_wav)")

    # TTS params (match your tts_client payload fields)
    ap.add_argument("--url", default="http://127.0.0.1:8000/tts")
    ap.add_argument("--voice-prompt", default=None)
    ap.add_argument("--cfg-scale", type=float, default=4.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.5)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--do-sample", action="store_true", help="Enable sampling")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--trim-start", type=float, default=0.0)
    ap.add_argument("--trim-end", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=600)

    # Language steering like your “Dobro jutro ... Dobro jutro”
    ap.add_argument("--prefix", default="Dobro jutro. ", help="Prefix added before each spoken item")
    ap.add_argument("--suffix", default=" Dobro jutro.", help="Suffix added after each spoken item")

    ap.add_argument("--overwrite-wavs", action="store_true", help="Re-generate even if wav already exists")
    ap.add_argument("--dry-run", action="store_true", help="Do not call TTS, only show what would be done")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    in_csv = Path(args.csv)
    if not in_csv.exists():
        print(f"ERROR: input CSV not found: {in_csv}", file=sys.stderr)
        return 2

    rows, delim = read_csv_any_delim(in_csv)
    if not rows:
        print("ERROR: CSV is empty.", file=sys.stderr)
        return 2

    out_csv = Path(args.out_csv) if args.out_csv else in_csv.with_name(in_csv.stem + "_with_tts.csv")
    audio_dir = Path(args.audio_dir) if args.audio_dir else in_csv.with_name(in_csv.stem + "_audio")
    ensure_dir(audio_dir)

    # Ensure wav column exists (append at end)
    header = rows[0]
    have_header = bool(args.skip_header)
    wav_col_idx: Optional[int] = None

    if have_header:
        # add new column if missing
        if args.wav_col_name in header:
            wav_col_idx = header.index(args.wav_col_name)
        else:
            header.append(args.wav_col_name)
            wav_col_idx = len(header) - 1
            rows[0] = header
    else:
        # no header mode: always append a new column at end
        wav_col_idx = len(rows[0])  # after current columns
        for i in range(len(rows)):
            rows[i] = rows[i] + [""]  # make room

    sess = requests.Session()

    start_i = 1 if have_header else 0
    for i in range(start_i, len(rows)):
        r = rows[i]

        # pad row to wav_col_idx
        if len(r) <= wav_col_idx:
            r.extend([""] * (wav_col_idx + 1 - len(r)))

        if args.col >= len(r):
            if args.verbose:
                print(f"Row {i}: column {args.col} missing, skipping")
            continue

        item = (r[args.col] or "").strip()
        if not item:
            if args.verbose:
                print(f"Row {i}: empty text, skipping")
            continue

        # deterministic filename: <row>-<slug>-<hash>.wav
        slug = safe_slug(item)
        h = sha1_short(item, 10)
        wav_name = f"{i:04d}_{slug}_{h}.wav"
        wav_path = audio_dir / wav_name

        rel_path = os.path.relpath(wav_path, start=out_csv.parent)

        # already there?
        if wav_path.exists() and not args.overwrite_wavs:
            r[wav_col_idx] = rel_path
            if args.verbose:
                print(f"Row {i}: exists -> {rel_path}")
            continue

        speak_text = f"{args.prefix}{item}{args.suffix}"

        if args.dry_run:
            r[wav_col_idx] = rel_path
            print(f"[DRY] Row {i}: '{item}' -> {wav_path}")
            continue

        if args.verbose:
            print(f"Row {i}: TTS '{item}' -> {wav_path}")

        try:
            tts_json = call_tts(
                url=args.url,
                text=speak_text,
                cfg_scale=args.cfg_scale,
                voice_prompt=args.voice_prompt,
                repetition_penalty=args.repetition_penalty,
                top_p=args.top_p,
                temperature=args.temperature,
                do_sample=args.do_sample,
                max_new_tokens=args.max_new_tokens,
                trim_start=args.trim_start,
                trim_end=args.trim_end,
                timeout_s=args.timeout,
            )
            materialize_audio(tts_json, wav_path, sess, timeout_s=args.timeout)
            r[wav_col_idx] = rel_path
        except Exception as e:
            # Write debugging JSON next to wav path
            dbg = audio_dir / f"{i:04d}_{slug}_{h}.json"
            try:
                dbg.write_text(json.dumps(tts_json if "tts_json" in locals() else {"error": str(e)}, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
            print(f"ERROR row {i} '{item}': {e}", file=sys.stderr)
            # keep going

        rows[i] = r

    write_csv(out_csv, rows, delim)
    print(f"Done. CSV written: {out_csv}")
    print(f"WAV dir: {audio_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
