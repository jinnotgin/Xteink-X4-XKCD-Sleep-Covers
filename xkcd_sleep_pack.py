#!/usr/bin/env python3
"""
xkcd_sleep_pack.py

Generates 480x800, 24-bit, uncompressed BMPs suitable for a device "sleep screen" folder.

Upgrades vs previous:
- Caches comic metadata + image dimensions in cache/meta.json
- Caches original downloaded images in cache/images/ to avoid re-downloading
- Subsequent runs typically only fetch the newest comics

Notes:
- This script can still do lots of requests on FIRST run. Subsequent runs are fast.
- Be polite to XKCD: keep a small delay.
- XKCD comics are CC BY-NC 2.5; this script writes CREDITS.txt.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image, ImageOps

from tqdm import tqdm

XKCD_LATEST_JSON = "https://xkcd.com/info.0.json"
XKCD_NUM_JSON = "https://xkcd.com/{num}/info.0.json"

TARGET_W = 480
TARGET_H = 800
TARGET_AR = TARGET_W / TARGET_H  # 0.6

DEFAULT_KEYS = [936, 927, 303, 1319, 1205, 323, 386, 353, 1179]

# Practical defaults (adaptive widening ensures we reach N)
DEFAULT_KEY_WEIGHT = 2.5
DEFAULT_KEY_MAX_DELTA = 0.28

DEFAULT_GENERAL_MAX_DELTA_START = 0.32
DEFAULT_GENERAL_MAX_DELTA_MAX = 0.70
DEFAULT_GENERAL_MAX_DELTA_STEP = 0.04

DEFAULT_MIN_SCALE_START = 0.42
DEFAULT_MIN_SCALE_MIN = 0.30
DEFAULT_MIN_SCALE_STEP = 0.02

DEFAULT_DELAY_S = 0.15  # be polite
DEFAULT_TIMEOUT_S = 25


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
    cached_image_path: Optional[str] = None


def slugify(text: str, max_len: int = 24) -> str:
    s = text.lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"\-+", "-", s).strip("-")
    if not s:
        s = "xkcd"
    return s[:max_len].rstrip("-")


def _sleep(delay_s: float) -> None:
    if delay_s > 0:
        time.sleep(delay_s)


def fetch_json(session: requests.Session, url: str, delay_s: float, timeout_s: float) -> Optional[dict]:
    try:
        resp = session.get(url, timeout=timeout_s)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        _sleep(delay_s)
        return resp.json()
    except Exception as e:
        print(f"[fetch_json error] {url}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def download_bytes(session: requests.Session, url: str, delay_s: float, timeout_s: float) -> Optional[bytes]:
    try:
        resp = session.get(url, timeout=timeout_s)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        _sleep(delay_s)
        return resp.content
    except Exception as e:
        print(f"[download error] {url}: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def guess_ext_from_url(url: str) -> str:
    # Basic, works for XKCD (usually png/jpg/gif)
    m = re.search(r"\.([a-zA-Z0-9]+)(?:\?|$)", url)
    if not m:
        return "img"
    ext = m.group(1).lower()
    if ext in {"png", "jpg", "jpeg", "gif", "webp"}:
        return ext
    return "img"


def read_image_dimensions(image_bytes: bytes) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = ImageOps.exif_transpose(im)
            w, h = im.size
            if w > 0 and h > 0:
                return w, h
    except Exception:
        return None
    return None


def render_to_bmp_from_path(src_path: Path, out_path: Path) -> None:
    with Image.open(src_path) as im:
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


def dedupe_by_num(items: List[Comic]) -> List[Comic]:
    seen = set()
    out = []
    for c in items:
        if c.num in seen:
            continue
        seen.add(c.num)
        out.append(c)
    return out


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
    # Stricter gating for keys
    key_candidates = [c for c in comics if c.is_key and c.fit_delta <= key_max_delta]

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
            best_pool = pool
            break

    def fit_score(c: Comic) -> float:
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


def estimate_bmp_bytes_per_image() -> int:
    # 24-bit BMP pixel bytes = w*h*3, plus header ~54 bytes.
    # Row padding is 0 because (480*3)=1440 divisible by 4.
    return TARGET_W * TARGET_H * 3 + 54


def load_cache(meta_path: Path) -> Dict[str, dict]:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        # expected: {"comics": {"123": {...}, ...}}
        return obj.get("comics", {}) if isinstance(obj, dict) else {}
    except Exception:
        return {}


def save_cache(meta_path: Path, comics_map: Dict[str, dict]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = meta_path.with_suffix(".tmp")
    data = {"comics": comics_map}
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp_path.replace(meta_path)


def build_comic_from_cached_entry(num: int, entry: dict, is_key: bool) -> Optional[Comic]:
    try:
        title = entry["title"]
        img_url = entry["img_url"]
        alt = entry.get("alt", "")
        w = int(entry["w"])
        h = int(entry["h"])
        cached_image_path = entry.get("image_path")

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
            cached_image_path=cached_image_path,
        )
    except Exception:
        return None


def ensure_comic_cached(
    session: requests.Session,
    num: int,
    is_key: bool,
    cache_map: Dict[str, dict],
    images_dir: Path,
    delay_s: float,
    timeout_s: float,
    refresh_cache: bool,
) -> Optional[Comic]:
    """
    Ensure we have stable metadata (title/img_url/alt) and dimensions for this comic,
    stored in cache_map. Optionally ensure original image is cached on disk.
    """

    key = str(num)
    entry = cache_map.get(key)

    if entry and not refresh_cache:
        # Validate minimal fields exist
        c = build_comic_from_cached_entry(num, entry, is_key)
        if c is not None:
            # If image cache path exists on disk, keep it
            if c.cached_image_path:
                p = Path(c.cached_image_path)
                if not p.is_file():
                    # stale path
                    entry.pop("image_path", None)
                    c.cached_image_path = None
            return c

    # Fetch JSON (needed for img_url/title)
    info = fetch_json(session, XKCD_NUM_JSON.format(num=num), delay_s, timeout_s)
    if info is None:
        return None

    img_url = info.get("img")
    if not img_url:
        return None

    title = info.get("title", f"xkcd-{num}")
    alt = info.get("alt", "")

    # If we have entry but img_url changed, we must refresh image/dims
    need_dims = True
    cached_image_path: Optional[str] = None

    if entry and not refresh_cache:
        if entry.get("img_url") == img_url and entry.get("w") and entry.get("h"):
            # dimensions ok; just reuse
            need_dims = False
            cached_image_path = entry.get("image_path")
            if cached_image_path and not Path(cached_image_path).is_file():
                cached_image_path = None
                entry.pop("image_path", None)

    w = h = None
    if need_dims:
        img_bytes = download_bytes(session, img_url, delay_s, timeout_s)
        if not img_bytes:
            return None
        dims = read_image_dimensions(img_bytes)
        if not dims:
            return None
        w, h = dims

        # Also cache original image bytes on disk
        ext = guess_ext_from_url(img_url)
        images_dir.mkdir(parents=True, exist_ok=True)
        img_path = images_dir / f"{num:04d}.{ext}"
        try:
            img_path.write_bytes(img_bytes)
            cached_image_path = str(img_path)
        except Exception:
            cached_image_path = None
    else:
        w = int(entry["w"])
        h = int(entry["h"])

    # Update cache entry
    cache_map[key] = {
        "num": num,
        "title": title,
        "alt": alt,
        "img_url": img_url,
        "w": int(w),
        "h": int(h),
        "image_path": cached_image_path,
        "cached_at_unix": int(time.time()),
    }

    return build_comic_from_cached_entry(num, cache_map[key], is_key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output directory (use your SD card /sleep folder).")
    ap.add_argument("--n", type=int, default=150, help="Number of images to output (default: 150).")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="Delay between requests (seconds).")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Request timeout (seconds).")

    ap.add_argument("--cache-dir", type=str, default="./cache",
                    help="Cache directory for meta.json and cached images (default: ./cache).")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Ignore cache and refetch metadata/dimensions.")

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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir)
    meta_path = cache_dir / "meta.json"
    images_dir = cache_dir / "images"

    # Parse key comics
    key_set = set()
    for part in args.keys.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            key_set.add(int(part))
        except ValueError:
            pass

    # Requests session
    session = requests.Session()
    session.headers["User-Agent"] = "Xteink-X4-XKCD-Sleep-Covers/1.1 (noncommercial, attribution; local script)"

    # Load cache
    cache_map = load_cache(meta_path)
    if args.refresh_cache:
        print("Cache refresh enabled: will refetch metadata/dimensions as needed.", file=sys.stderr)

    # Fetch latest
    latest = fetch_json(session, XKCD_LATEST_JSON, args.delay, args.timeout)
    if not latest or "num" not in latest:
        print(f"Failed to fetch latest XKCD JSON from {XKCD_LATEST_JSON}", file=sys.stderr)
        print("Try: curl -I https://xkcd.com/info.0.json (to check network/proxy/captive portal)", file=sys.stderr)
        return 2

    max_num = int(latest["num"])
    print(f"Latest XKCD num: {max_num}")

    comics: List[Comic] = []
    missing = 0
    analyzed = 0
    updated_cache = False

    # Iterate all comics; use cache to avoid re-downloading images just for dimensions
    for num in tqdm(range(1, max_num + 1), desc="Scanning XKCD", unit="comic"):
        c = ensure_comic_cached(
            session=session,
            num=num,
            is_key=(num in key_set),
            cache_map=cache_map,
            images_dir=images_dir,
            delay_s=args.delay,
            timeout_s=args.timeout,
            refresh_cache=args.refresh_cache,
        )
        if c is None:
            missing += 1
        else:
            comics.append(c)
            analyzed += 1
            updated_cache = True

        # Periodically save cache so you don’t lose progress on long runs
        if num % 200 == 0 and updated_cache:
            save_cache(meta_path, cache_map)
            updated_cache = False
            print(f"Progress: {num}/{max_num} (analyzed {analyzed}, missing {missing})")

    if updated_cache:
        save_cache(meta_path, cache_map)

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

    # Remove previously generated BMPs matching our pattern
    for fn in out_dir.iterdir():
        if fn.is_file() and fn.suffix.lower() == ".bmp" and re.match(r"^\d{3}__xkcd\d{4}__", fn.name):
            try:
                fn.unlink()
            except OSError:
                pass

    credits_lines: List[str] = []
    print(f"Selection meta: {meta}")

    # Render selected comics
    for idx, c in enumerate(tqdm(selected, desc="Rendering BMPs", unit="img"), start=1):
        rank = idx
        delta_tag = int(round(c.fit_delta * 1000))
        slug = slugify(c.title)
        fname = f"{rank:03d}__xkcd{c.num:04d}__d{delta_tag:03d}__{slug}.bmp"
        out_path = out_dir / fname

        # Prefer cached original image file
        src_path: Optional[Path] = None
        if c.cached_image_path:
            p = Path(c.cached_image_path)
            if p.is_file():
                src_path = p

        if src_path is None:
            # Download + cache original image
            img_bytes = download_bytes(session, c.img_url, args.delay, args.timeout)
            if not img_bytes:
                print(f"Warning: failed to download image for xkcd {c.num}, skipping.", file=sys.stderr)
                continue
            ext = guess_ext_from_url(c.img_url)
            images_dir.mkdir(parents=True, exist_ok=True)
            src_path = images_dir / f"{c.num:04d}.{ext}"
            try:
                src_path.write_bytes(img_bytes)
                # Update cache path for next run
                cache_map[str(c.num)]["image_path"] = str(src_path)
            except Exception:
                # If we can't write cache image, render directly from bytes via a temp file-like path not supported;
                # simplest: write to out_dir as temp.
                tmp = out_dir / f".tmp_{c.num:04d}.{ext}"
                tmp.write_bytes(img_bytes)
                src_path = tmp

        try:
            render_to_bmp_from_path(src_path, out_path)
        except Exception as e:
            print(f"Warning: failed to render xkcd {c.num}: {e}", file=sys.stderr)
            continue
        finally:
            # cleanup temp
            if src_path.name.startswith(".tmp_"):
                try:
                    src_path.unlink()
                except OSError:
                    pass

        credits_lines.append(
            f"{rank:03d}\txkcd {c.num}\t{c.title}\t(ar={c.ar:.3f}, delta={c.fit_delta:.3f}, scale={c.scale_to_fit:.3f})\t{c.img_url}"
        )

    # Save cache after rendering (paths may have been updated)
    save_cache(meta_path, cache_map)

    # Write credits for attribution
    credits_path = out_dir / "CREDITS.txt"
    with credits_path.open("w", encoding="utf-8") as f:
        f.write("XKCD Sleep Pack\n")
        f.write("License: XKCD comics are CC BY-NC 2.5. Please attribute Randall Munroe / xkcd.com.\n")
        f.write("Generated files are resized/padded to 480x800 BMP for a device sleep screen.\n\n")
        f.write("Rank\tComic\tTitle\tInfo\tImageURL\n")
        f.write("\n".join(credits_lines))
        f.write("\n")

    # Size estimate
    per_img = estimate_bmp_bytes_per_image()
    total = per_img * len(credits_lines)
    print(f"Estimated size: ~{per_img} bytes/image; ~{total} bytes total for {len(credits_lines)} images.")
    print(f"Output written to: {out_dir}")
    print(f"Cache stored in: {cache_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
