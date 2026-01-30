#!/usr/bin/env python3
"""
xkcd_sleep_pack.py

Generates 480x800, 24-bit, uncompressed BMPs suitable for a device "sleep screen" folder.

Key features:
- Downloads all XKCD comics up to current (auto-detects latest num)
- Scores comics by aspect-ratio closeness to portrait (target AR = 0.6)
- Preferentially includes configurable "key" comics, BUT excludes key comics with bad ratios
- Selects top N unique comics (no duplication/multiplicity weighting)
- Writes ranked filenames: rank__xkcdNNNN__dDDD__slug.bmp
- Writes a CREDITS.txt file for attribution

Notes:
- This script makes many requests. It includes a small delay and a User-Agent.
- Noncommercial use + attribution required by XKCD license.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageOps

XKCD_LATEST_JSON = "https://xkcd.com/info.0.json"
XKCD_NUM_JSON = "https://xkcd.com/{num}/info.0.json"

TARGET_W = 480
TARGET_H = 800
TARGET_AR = TARGET_W / TARGET_H  # 0.6


DEFAULT_KEYS = [936, 927, 303, 1319, 1205, 323, 386, 353, 1179]

# Good practical defaults (adaptive widening will ensure we reach N)
DEFAULT_KEY_WEIGHT = 2.5
DEFAULT_KEY_MAX_DELTA = 0.28

DEFAULT_GENERAL_MAX_DELTA_START = 0.32
DEFAULT_GENERAL_MAX_DELTA_MAX = 0.70
DEFAULT_GENERAL_MAX_DELTA_STEP = 0.04

DEFAULT_MIN_SCALE_START = 0.42
DEFAULT_MIN_SCALE_MIN = 0.30
DEFAULT_MIN_SCALE_STEP = 0.02

DEFAULT_DELAY_S = 0.15  # be polite


@dataclass
class Comic:
    num: int
    title: str
    img_url: str
    alt: str
    w: int
    h: int
    ar: float
    fit_delta: float
    scale_to_fit: float
    is_key: bool


def slugify(text: str, max_len: int = 24) -> str:
    s = text.lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"\-+", "-", s).strip("-")
    if not s:
        s = "xkcd"
    return s[:max_len].rstrip("-")


def fetch_json(session: requests.Session, url: str, delay_s: float) -> Optional[dict]:
    try:
        resp = session.get(url, timeout=20)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(delay_s)
        return resp.json()
    except Exception:
        return None


def download_image(session: requests.Session, url: str, delay_s: float) -> Optional[bytes]:
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        time.sleep(delay_s)
        return resp.content
    except Exception:
        return None


def analyze_comic(session: requests.Session, info: dict, is_key: bool, delay_s: float) -> Optional[Comic]:
    num = int(info.get("num"))
    title = info.get("title", f"xkcd-{num}")
    img_url = info.get("img")
    alt = info.get("alt", "")

    if not img_url:
        return None

    img_bytes = download_image(session, img_url, delay_s)
    if not img_bytes:
        return None

    try:
        with Image.open(io.BytesIO(img_bytes)) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
    except Exception:
        return None

    if w <= 0 or h <= 0:
        return None

    ar = w / h
    fit_delta = abs(ar - TARGET_AR)
    scale_to_fit = min(TARGET_W / w, TARGET_H / h)

    return Comic(
        num=num,
        title=title,
        img_url=img_url,
        alt=alt,
        w=w,
        h=h,
        ar=ar,
        fit_delta=fit_delta,
        scale_to_fit=scale_to_fit,
        is_key=is_key,
    )


def render_to_bmp(image_bytes: bytes, out_path: str) -> None:
    with Image.open(io.BytesIO(image_bytes)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")

        w, h = im.size
        scale = min(TARGET_W / w, TARGET_H / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        im_resized = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
        x = (TARGET_W - new_w) // 2
        y = (TARGET_H - new_h) // 2
        canvas.paste(im_resized, (x, y))

        # BMP saved from RGB at this size is 24-bit and uncompressed by default in Pillow.
        canvas.save(out_path, format="BMP")


def adaptive_select(
    comics: List[Comic],
    n: int,
    key_weight: float,
    key_max_delta: float,
    general_delta_start: float,
    general_delta_max: float,
    general_delta_step: float,
    min_scale_start: float,
    min_scale_min: float,
    min_scale_step: float,
) -> Tuple[List[Comic], Dict[str, float]]:
    """
    Select top-n comics with adaptive constraints. No duplicates.
    Keys are preferred via score multiplier, but keys must pass key_max_delta.
    """

    # Pre-split keys for stricter gating.
    key_candidates = [c for c in comics if c.is_key and c.fit_delta <= key_max_delta]

    # We'll widen the general threshold and relax min_scale until enough candidates.
    general_delta = general_delta_start
    min_scale = min_scale_start

    best_pool: List[Comic] = []

    while True:
        gen_candidates = [
            c for c in comics
            if (not c.is_key)
            and c.fit_delta <= general_delta
            and c.scale_to_fit >= min_scale
        ]
        key_ok = [c for c in key_candidates if c.scale_to_fit >= min_scale]

        pool = dedupe_by_num(key_ok + gen_candidates)

        if len(pool) >= n:
            best_pool = pool
            break

        widened = False
        if general_delta + 1e-9 < general_delta_max:
            general_delta = min(general_delta_max, general_delta + general_delta_step)
            widened = True
        if (not widened) and (min_scale - 1e-9 > min_scale_min):
            min_scale = max(min_scale_min, min_scale - min_scale_step)
            widened = True

        if not widened:
            # Can't widen anymore; take what we have.
            best_pool = pool
            break

    # Score and take top-n
    # Fit score normalized by final general_delta used.
    def fit_score(c: Comic) -> float:
        # If someone is just under the delta threshold -> near 0; perfect -> 1.
        return max(0.0, 1.0 - (c.fit_delta / max(1e-9, general_delta)))

    def adj_score(c: Comic) -> float:
        base = fit_score(c)
        return base * (key_weight if c.is_key else 1.0)

    best_pool.sort(
        key=lambda c: (
            -adj_score(c),
            c.fit_delta,
            -c.scale_to_fit,
            c.num,
        )
    )

    selected = best_pool[:n]

    meta = {
        "general_delta_final": general_delta,
        "min_scale_final": min_scale,
        "key_max_delta": key_max_delta,
        "key_weight": key_weight,
        "pool_size": float(len(best_pool)),
        "selected_size": float(len(selected)),
    }
    return selected, meta


def dedupe_by_num(items: List[Comic]) -> List[Comic]:
    seen = set()
    out = []
    for c in items:
        if c.num in seen:
            continue
        seen.add(c.num)
        out.append(c)
    return out


def estimate_bmp_bytes_per_image() -> int:
    # 24-bit BMP pixel bytes = w*h*3, plus header ~54 bytes.
    # Row padding is 0 because (480*3)=1440 divisible by 4.
    return TARGET_W * TARGET_H * 3 + 54


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (use your SD card /sleep folder).")
    ap.add_argument("--n", type=int, default=150, help="Number of images to output (default: 150).")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="Delay between requests (seconds).")
    ap.add_argument("--key_weight", type=float, default=DEFAULT_KEY_WEIGHT)
    ap.add_argument("--key_max_delta", type=float, default=DEFAULT_KEY_MAX_DELTA)
    ap.add_argument("--general_delta_start", type=float, default=DEFAULT_GENERAL_MAX_DELTA_START)
    ap.add_argument("--general_delta_max", type=float, default=DEFAULT_GENERAL_MAX_DELTA_MAX)
    ap.add_argument("--general_delta_step", type=float, default=DEFAULT_GENERAL_MAX_DELTA_STEP)
    ap.add_argument("--min_scale_start", type=float, default=DEFAULT_MIN_SCALE_START)
    ap.add_argument("--min_scale_min", type=float, default=DEFAULT_MIN_SCALE_MIN)
    ap.add_argument("--min_scale_step", type=float, default=DEFAULT_MIN_SCALE_STEP)
    ap.add_argument("--keys", type=str, default=",".join(str(x) for x in DEFAULT_KEYS),
                    help="Comma-separated XKCD numbers to treat as key comics.")
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    key_set = set()
    for part in args.keys.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            key_set.add(int(part))
        except ValueError:
            pass

    session = requests.Session()
    session.headers["User-Agent"] = "xkcd-sleep-pack/1.0 (noncommercial, attribution; contact: local-script)"

    latest = fetch_json(session, XKCD_LATEST_JSON, args.delay)
    if not latest or "num" not in latest:
        print("Failed to fetch latest XKCD JSON.", file=sys.stderr)
        return 2

    max_num = int(latest["num"])
    print(f"Latest XKCD num: {max_num}")

    comics: List[Comic] = []
    missing = 0

    for num in range(1, max_num + 1):
        info = fetch_json(session, XKCD_NUM_JSON.format(num=num), args.delay)
        if info is None:
            missing += 1
            continue
        c = analyze_comic(session, info, is_key=(num in key_set), delay_s=args.delay)
        if c:
            comics.append(c)

        # light progress
        if num % 250 == 0:
            print(f"Processed up to {num}/{max_num} (kept {len(comics)} comics, missing {missing})")

    print(f"Total analyzed comics: {len(comics)} (missing pages: {missing})")

    selected, meta = adaptive_select(
        comics=comics,
        n=args.n,
        key_weight=args.key_weight,
        key_max_delta=args.key_max_delta,
        general_delta_start=args.general_delta_start,
        general_delta_max=args.general_delta_max,
        general_delta_step=args.general_delta_step,
        min_scale_start=args.min_scale_start,
        min_scale_min=args.min_scale_min,
        min_scale_step=args.min_scale_step,
    )

    if not selected:
        print("No comics selected. Try relaxing thresholds.", file=sys.stderr)
        return 3

    # Clear existing BMPs in out_dir (only those matching our pattern), optional safety:
    # You can comment this out if you prefer.
    for fn in os.listdir(out_dir):
        if fn.lower().endswith(".bmp") and re.match(r"^\d{3}__xkcd\d{4}__", fn):
            try:
                os.remove(os.path.join(out_dir, fn))
            except OSError:
                pass

    # Download and render selected comics
    credits_lines = []
    print(f"Selection meta: {meta}")

    for idx, c in enumerate(selected, start=1):
        rank = idx
        delta_tag = int(round(c.fit_delta * 1000))
        slug = slugify(c.title)
        fname = f"{rank:03d}__xkcd{c.num:04d}__d{delta_tag:03d}__{slug}.bmp"
        out_path = os.path.join(out_dir, fname)

        img_bytes = download_image(session, c.img_url, args.delay)
        if not img_bytes:
            print(f"Warning: failed to download image for xkcd {c.num}, skipping.")
            continue

        try:
            render_to_bmp(img_bytes, out_path)
        except Exception as e:
            print(f"Warning: failed to render xkcd {c.num}: {e}")
            continue

        credits_lines.append(
            f"{rank:03d}\txkcd {c.num}\t{c.title}\t(ar={c.ar:.3f}, delta={c.fit_delta:.3f}, scale={c.scale_to_fit:.3f})\t{c.img_url}"
        )

    # Write credits for attribution
    credits_path = os.path.join(out_dir, "CREDITS.txt")
    with open(credits_path, "w", encoding="utf-8") as f:
        f.write("XKCD Sleep Pack\n")
        f.write("License: XKCD comics are CC BY-NC 2.5. Please attribute Randall Munroe / xkcd.com.\n")
        f.write("Generated files are resized/padded to 480x800 BMP for a device sleep screen.\n\n")
        f.write("Rank\tComic\tTitle\tInfo\tImageURL\n")
        f.write("\n".join(credits_lines))
        f.write("\n")

    # Size estimate
    per_img = estimate_bmp_bytes_per_image()
    total = per_img * len([ln for ln in credits_lines])
    print(f"Estimated size: ~{per_img} bytes/image; ~{total} bytes total for {len(credits_lines)} images.")
    print(f"Output written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
