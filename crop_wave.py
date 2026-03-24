#!/usr/bin/env python3
"""
Crop a WAV file to the region between the *first* and *last* silent pause.

Definition used:
- "silent pause" = a contiguous span of at least --min-silence-ms where the
  short-time RMS is below --threshold-db (relative to full scale, dBFS).

Output:
- Writes a new WAV that starts at the end of the first detected silent pause
  and ends at the start of the last detected silent pause.

Notes:
- Works with PCM WAV (8/16/24/32-bit), mono or multi-channel.
- If fewer than 2 pauses are found, it falls back to trimming leading/trailing silence.
"""

import argparse
import wave
import numpy as np
from pathlib import Path


def read_wav(path: str):
    with wave.open(path, "rb") as wf:
        params = wf.getparams()  # nchannels, sampwidth, framerate, nframes, comptype, compname
        raw = wf.readframes(params.nframes)

    nch = params.nchannels
    sw = params.sampwidth  # bytes per sample
    sr = params.framerate

    if params.comptype != "NONE":
        raise ValueError(f"Compressed WAV not supported (comptype={params.comptype}). Use PCM WAV.")

    # Convert raw bytes -> numpy array shaped (nframes, nchannels)
    if sw == 1:
        # 8-bit PCM in WAV is unsigned
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        x = (x - 128)  # center to signed
    elif sw == 2:
        x = np.frombuffer(raw, dtype=np.int16)
    elif sw == 3:
        # 24-bit little-endian PCM: unpack to int32
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b.reshape(-1, 3)
        x = (b[:, 0].astype(np.int32)
             | (b[:, 1].astype(np.int32) << 8)
             | (b[:, 2].astype(np.int32) << 16))
        # sign extend
        x = (x ^ 0x800000) - 0x800000
    elif sw == 4:
        x = np.frombuffer(raw, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported sample width: {sw} bytes/sample")

    if x.size % nch != 0:
        raise ValueError("WAV data size not divisible by channel count (corrupt file?)")

    x = x.reshape(-1, nch)
    return params, sr, x


def write_wav(path: str, params, frames: np.ndarray):
    nch = params.nchannels
    sw = params.sampwidth

    if frames.ndim != 2 or frames.shape[1] != nch:
        raise ValueError("frames must have shape (nframes, nchannels)")

    # Convert numpy array back to bytes
    if sw == 1:
        y = np.clip(frames[:, :], -128, 127).astype(np.int16)
        y = (y + 128).astype(np.uint8)
        raw = y.tobytes()
    elif sw == 2:
        y = np.clip(frames, -32768, 32767).astype(np.int16)
        raw = y.tobytes()
    elif sw == 3:
        y = np.clip(frames, -8388608, 8388607).astype(np.int32)
        # pack little-endian 24-bit
        b0 = (y & 0xFF).astype(np.uint8)
        b1 = ((y >> 8) & 0xFF).astype(np.uint8)
        b2 = ((y >> 16) & 0xFF).astype(np.uint8)
        raw = np.stack([b0, b1, b2], axis=-1).reshape(-1).tobytes()
    elif sw == 4:
        y = np.clip(frames, -2147483648, 2147483647).astype(np.int32)
        raw = y.tobytes()
    else:
        raise ValueError(f"Unsupported sample width: {sw} bytes/sample")

    with wave.open(path, "wb") as wf:
        wf.setnchannels(params.nchannels)
        wf.setsampwidth(params.sampwidth)
        wf.setframerate(params.framerate)
        wf.writeframes(raw)


def rms_dbfs(window: np.ndarray, full_scale: float) -> float:
    # window: (nframes, nch)
    v = window.astype(np.float64)

    # Remove DC offset per channel (important!)
    v = v - np.mean(v, axis=0, keepdims=True)

    # RMS per channel, then take the max channel energy
    ms_ch = np.mean(v * v, axis=0)
    rms_ch = np.sqrt(ms_ch) + 1e-12
    rms = float(np.max(rms_ch))

    return 20.0 * np.log10(rms / full_scale)



def find_silent_runs(mask: np.ndarray, min_len: int):
    """
    mask: boolean array length N, True where silent.
    Returns list of (start_idx, end_idx) inclusive-exclusive runs where silent and length>=min_len.
    """
    runs = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        if (j - i) >= min_len:
            runs.append((i, j))
        i = j
    return runs


def main():
    ap = argparse.ArgumentParser(description="Crop a WAV between first and last silent pause.")
    ap.add_argument("input", help="Input WAV file")
    ap.add_argument("-o", "--output", help="Output WAV file (default: <input>_cropped.wav)")
    ap.add_argument("--window-ms", type=float, default=20.0, help="Analysis window size in ms (default: 20)")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="Hop size in ms (default: 10)")
    ap.add_argument("--threshold-db", type=float, default=-40.0,
                    help="Silence threshold in dBFS (default: -40). More negative = stricter silence.")
    ap.add_argument("--min-silence-ms", type=float, default=300.0,
                    help="Minimum duration (ms) to count as a silent pause (default: 300)")
    ap.add_argument("--keep-pause-ms", type=float, default=200.0,
                help="How much silence of the detected pauses to keep (default: 200ms)")
                
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output
    if not out_path:
        p = Path(in_path)
        out_path = str(p.with_name(p.stem + "_cropped" + p.suffix))

    params, sr, x = read_wav(in_path)

    sw = params.sampwidth
    if sw == 1:
        full_scale = 127.0
    elif sw == 2:
        full_scale = 32767.0
    elif sw == 3:
        full_scale = 8388607.0
    elif sw == 4:
        full_scale = 2147483647.0
    else:
        raise ValueError("Unsupported sample width")

    win = max(1, int(round(sr * args.window_ms / 1000.0)))
    hop = max(1, int(round(sr * args.hop_ms / 1000.0)))
    min_sil_len = max(1, int(round(args.min_silence_ms / args.hop_ms)))  # in hops/windows

    # Compute silence mask per hop-window
    nframes = x.shape[0]
    if nframes < win:
        raise ValueError("Audio is shorter than the analysis window.")

    dbs = []
    positions = []  # start sample for each window
    for start in range(0, nframes - win + 1, hop):
        w = x[start:start + win]
        db = rms_dbfs(w, full_scale)
        dbs.append(db)
        positions.append(start)

    dbs = np.array(dbs)
    silent = dbs < args.threshold_db

    runs = find_silent_runs(silent, min_sil_len)
    nwin = len(silent)

    def sample_at_window_index(k: int) -> int:
        return positions[k]

    def is_leading(run):
        return run[0] == 0

    def is_trailing(run):
        return run[1] == nwin  # run reaches the very end of the analysis windows

    if not runs:
        # No pauses found: trim leading/trailing silence by finding first/last non-silent window
        non = np.where(~silent)[0]
        if non.size == 0:
            crop_start, crop_end = 0, nframes
        else:
            first_non = int(non[0])
            last_non = int(non[-1])
            crop_start = sample_at_window_index(first_non)
            crop_end = sample_at_window_index(last_non) + win
    else:
        # --- start: end of the first pause (prefer leading if present) ---
        lead_tol_ms = 350.0  # allow the "leading pause" to start within first 200ms
        lead_tol_win = int(round(lead_tol_ms / args.hop_ms))

        if runs[0][0] == 0:
            del runs[0]
        leading_runs = [r for r in runs if r[0] <= lead_tol_win]
        if leading_runs:
            first_pause = leading_runs[0]
            crop_start = sample_at_window_index(first_pause[1] - 1) + win
        else:
            # no leading pause detected -> don't cut based on an interior pause
            crop_start = 0

        # --- end: start of the last pause, but ignore trailing silence ---
        last_pause = runs[-1]
        if is_trailing(last_pause) and len(runs) >= 2:
            last_pause = runs[-2]  # use the pause before the trailing silence
        crop_end = sample_at_window_index(last_pause[0])

    # clamp + sanity
    crop_start = int(np.clip(crop_start, 0, nframes))
    crop_end = int(np.clip(crop_end, 0, nframes))
    
    keep = int(round(sr * args.keep_pause_ms / 1000.0))

    crop_start = max(0, crop_start - keep)   # include part of first pause
    crop_end   = min(nframes, crop_end + keep)  # include part of last pause
    
    if crop_end <= crop_start:
        crop_start, crop_end = 0, nframes

    y = x[crop_start:crop_end].copy()
    write_wav(out_path, params, y)

    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    print(f"Cropped samples: {crop_start} .. {crop_end} (duration {((crop_end - crop_start)/sr):.3f}s)")
    print(f"Params: {params.nchannels}ch, {params.sampwidth*8}-bit, {sr}Hz")


if __name__ == "__main__":
    main()
