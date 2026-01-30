# Xteink X4 XKCD Sleep Covers

Generate **portrait-friendly XKCD comics** as **480×800, 24-bit, uncompressed BMP** images for **[Xteink X4](https://www.xteink.com/products/xteink-x4)** devices running the **[CrossPoint Reader](https://github.com/crosspoint-reader/crosspoint-reader)** firmware.

This repo:
- Scans XKCD **up to the current comic**
- Scores each comic by how well it fits a **480×800 portrait screen**
- Preferentially includes configurable **key comics** (only if they fit well enough)
- Exports a ranked set of **unique** BMPs into a `/sleep/` folder, ready to copy to your SD card
- Uses **caching** to avoid re-downloading stable metadata/images on later runs
- Shows progress with **tqdm**

---

## Device SD card setup

Your device supports two modes:

- **Single image**: put `sleep.bmp` in the SD card root
- **Multiple images (recommended)**: create a `sleep/` directory in the SD card root and place `.bmp` files inside  
  If `sleep/` exists, the device will randomly pick one BMP whenever it sleeps.

This project generates the **multi-image** folder:

```text
SD_CARD_ROOT/
  sleep/
    001__xkcd1205__d018__is-it-worth-the-time.bmp
    002__xkcd0936__d041__password-strength.bmp
    ...
    CREDITS.txt
````

> Make sure your device’s **Sleep Screen** setting is set to **Custom**.

---

## Output format & file naming

* Output: **BMP**, **24-bit RGB**, **uncompressed**
* Resolution: **480×800**
* Padding is used (white background) to preserve the comic’s aspect ratio without cropping.
* Optional footer bar and top/bottom padding reserve space while still outputting **480×800**.

Filenames are ranked for easy browsing:

```text
{rank}__xkcd{num}__d{delta}__{slug}.bmp
```

* `rank`: `001..100` (sortable)
* `xkcd####`: the comic number (zero-padded)
* `d###`: fit delta × 1000 (smaller = closer to portrait ratio)
* `slug`: sanitized title (optional but helpful)

---

## Selection logic (how “good fits” are chosen)

* Target portrait aspect ratio: `480/800 = 0.6`
* Each comic image is scored by:

```text
fit_delta = abs((width / height) - 0.6)
```

Lower `fit_delta` = better portrait fit.

### Key comics (preferred, but guarded)

You can configure a list of “key” comic IDs. They are **preferred** during ranking, but they must pass a stricter fit threshold.
Bad-ratio key comics are automatically skipped.

---

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Where `requirements.txt` contains:

```text
requests
Pillow
tqdm
```

---

## Usage

Generate the default top 100 into a local folder:

```bash
python xkcd_sleep_pack.py --out ./out/sleep --n 100
```

Then copy the entire `sleep/` folder to the **root of your SD card**.

### Useful flags

* `--n` — number of images (default 100)
* `--cache-dir` — cache folder path (default `./cache`)
* `--refresh-cache` — ignore cache and rebuild metadata/dimensions
* `--keys` — comma-separated key comic IDs
* `--key_weight` — how strongly keys are preferred in ranking (no duplicate files)
* `--workers` — parallel download worker threads
* `--rps` — **global** HTTP request cap (requests/sec) across all threads
* `--footer` — add a small footer bar with `xkcd # — title`
* `--footer-h` — footer height in pixels (default 24)
* `--footer-font` — footer font size (default 16)
* `--footer-bg` — footer background color (default `#FFFFFF`)
* `--pad-top` — top padding in pixels (default 8)
* `--pad-bottom` — bottom padding in pixels (default 4)

Example (faster but polite):

```bash
python xkcd_sleep_pack.py \
  --out ./out/sleep \
  --n 150 \
  --workers 8 \
  --rps 4 \
  --keys 936,927,303,1319,1205,323,386,353,1179 \
  --key_weight 2.5 \
  --footer \
  --pad-top 8 \
  --pad-bottom 4
```

---

## Footer & Padding behavior

* The footer and padding reserve vertical space; the comic is centered in the remaining content region.
* Selection thresholds use the **content height** (after footer/padding) for `scale_to_fit`, so tall comics are filtered consistently with your chosen layout.

---

## Caching notes

The script caches stable metadata and downloaded originals:

```text
cache/
  meta.json
  images/
    0001.png
    0002.png
    ...
```

After the first run, subsequent runs are typically much faster because only new comics need to be fetched.

---

## GitHub Actions

You can run this via GitHub Actions to produce a downloadable ZIP artifact (handy if you don’t want to run Python locally).
The workflow also caches `cache/` between runs to speed up scheduled builds.

See: `.github/workflows/build.yml`

---

## License & Attribution

XKCD comics are licensed under **Creative Commons Attribution–NonCommercial 2.5**.

This tool downloads comics and generates resized/padded BMPs for personal/noncommercial use. It also writes a `CREDITS.txt` file for attribution.

* XKCD JSON API: [https://xkcd.com/json.html](https://xkcd.com/json.html)
* XKCD license: [https://xkcd.com/license.html](https://xkcd.com/license.html)

If you redistribute generated packs, ensure you comply with the XKCD license (attribution + noncommercial).

---

## Disclaimer

This project is not affiliated with XKCD. XKCD is © Randall Munroe.
