"""
VisionSync — COCO Replay Builder

ရည်ရွယ်ချက်
-----------
Class အသစ်ထည့်ပြီး fine-tune လုပ်တဲ့အခါ မူလ COCO ၈၀ မျိုးလည်း ဆက်အလုပ်လုပ်နေစေရန်။

ဘာကြောင့် လိုအပ်လဲ
------------------
Ultralytics မှာ base model ရဲ့ `nc` နဲ့ dataset ရဲ့ `nc` မတူတာနဲ့ detection head ကို
အသစ်ပြန်တည်ဆောက်သည် (`Overriding model.yaml nc=80 with nc=81`)။ ထို့ကြောင့် မူလ
class ၈၀ ရဲ့ weight တွေ ပျက်သွားပြီး၊ dataset ထဲမှာ အဲဒီ class တွေရဲ့ ပုံမပါရင်
model က ဘယ်တော့မှ ပြန်မမှတ်မိတော့ပါ (catastrophic forgetting)။

ဖြေရှင်းနည်း — **Replay**
COCO ရဲ့ ပုံအချို့ကို master dataset ထဲ ပြန်ထည့်ပေးထားလိုက်ရင် fine-tune လုပ်တဲ့အခါ
model က class အသစ်ရော အဟောင်း ၈၀ ရော တစ်ပြိုင်တည်း သင်ယူသွားမည်။

Usage (CLI):
    # အမြန်စမ်းရန် (၇ MB၊ ပုံ ၁၂၈ ပုံ)
    python coco_replay.py --source coco128

    # တကယ်သုံးရန် (val2017 ~၈၀၀ MB၊ class ၈၀ လုံး ကောင်းစွာပါ)
    python coco_replay.py --source val2017 --per-class 30

    # အရင်ထည့်ထားတာကို ဖျက်ပြီး အသစ်ထည့်
    python coco_replay.py --source val2017 --per-class 50 --replace

Usage (API):
    POST /master/add-coco-replay   {"source": "val2017", "per_class": 30}
    GET  /master/replay-status
"""

import json
import os
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from master_builder import (
    BACKEND_DIR,
    COCO_NAMES,
    MASTER_DIR,
    _ensure_master_structure,
    _load_yaml,
    _normalize_names,
)

CACHE_DIR = os.path.join(BACKEND_DIR, ".cache", "coco")

# ပုံများကို master ထဲ ကူးတဲ့အခါ ဒီ prefix နဲ့ ကူးမည် — နောက်ပိုင်း ဖျက်ရလွယ်စေရန်
REPLAY_PREFIX = "cocoreplay_"
REPLAY_MARKER = ".coco_replay.json"

SOURCES: Dict[str, Dict[str, Any]] = {
    "coco128": {
        "label": "COCO128 (ပုံ ၁၂၈၊ ~၇ MB — အမြန်စမ်းရန်)",
        "urls": [("coco128.zip", "https://ultralytics.com/assets/coco128.zip")],
        "split": "train2017",
        "approx_mb": 7,
    },
    "val2017": {
        "label": "COCO val2017 (ပုံ ၅၀၀၀၊ ~၈၀၀ MB — class ၈၀ လုံးအတွက် အကောင်းဆုံး)",
        "urls": [
            ("coco2017labels.zip",
             "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip"),
            ("val2017.zip", "http://images.cocodataset.org/zips/val2017.zip"),
        ],
        "split": "val2017",
        "approx_mb": 830,
    },
}


def _noop(_: str) -> None:
    pass


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download(url: str, dest: str, on_log: Callable[[str], None]) -> None:
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        on_log(f"  ✔ cache ထဲ ရှိပြီးသား: {os.path.basename(dest)} "
               f"({os.path.getsize(dest) / 1024 / 1024:.1f} MB)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    on_log(f"  ⬇ Download: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "VisionSync/1.0"})
    last_pct = -10
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                pct = int(got * 100 / total)
                if pct - last_pct >= 10:
                    last_pct = pct
                    on_log(f"    {pct}%  ({got / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MB)")
    os.replace(tmp, dest)
    on_log(f"  ✔ ပြီးပါပြီ: {os.path.basename(dest)}")


def _extract(zip_path: str, out_dir: str, on_log: Callable[[str], None]) -> None:
    marker = os.path.join(out_dir, "." + os.path.basename(zip_path) + ".done")
    if os.path.isfile(marker):
        on_log(f"  ✔ extract ပြီးသား: {os.path.basename(zip_path)}")
        return
    os.makedirs(out_dir, exist_ok=True)
    on_log(f"  📦 Extract: {os.path.basename(zip_path)}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for m in zf.namelist():
            # Zip-slip ကာကွယ်မှု
            target = os.path.normpath(os.path.join(out_dir, m))
            if not target.startswith(os.path.normpath(out_dir) + os.sep) and target != os.path.normpath(out_dir):
                continue
            if m.endswith("/"):
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with zf.open(m) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    with open(marker, "w") as f:
        f.write("ok")


def _find_dir(root: str, name: str, parent_hint: Optional[str] = None) -> Optional[str]:
    """root အောက်မှာ `name` ဆိုတဲ့ folder ကို ရှာပေးမည် (parent_hint ပါရင် ဦးစားပေး)။"""
    matches: List[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d == name:
                matches.append(os.path.join(dirpath, d))
    if not matches:
        return None
    if parent_hint:
        for m in matches:
            if os.path.basename(os.path.dirname(m)) == parent_hint:
                return m
    return matches[0]


def _prepare_source(source: str, on_log: Callable[[str], None]) -> Tuple[str, str]:
    """Download + extract ပြီး (images_dir, labels_dir) ကို ပြန်ပေးမည်။"""
    spec = SOURCES[source]
    work = os.path.join(CACHE_DIR, source)
    os.makedirs(work, exist_ok=True)

    for fname, url in spec["urls"]:
        zp = os.path.join(CACHE_DIR, fname)
        _download(url, zp, on_log)
        _extract(zp, work, on_log)

    split = spec["split"]
    images_dir = _find_dir(work, split, parent_hint="images")
    labels_dir = _find_dir(work, split, parent_hint="labels")

    # val2017.zip က `val2017/` ကို တိုက်ရိုက်ထုတ်သည် (images parent မရှိ)
    if images_dir is None:
        cand = os.path.join(work, split)
        if os.path.isdir(cand):
            images_dir = cand
    if images_dir is None or labels_dir is None:
        raise RuntimeError(
            f"Extract ပြီးပေမယ့် images/labels folder မတွေ့ရှိပါ "
            f"(images={images_dir}, labels={labels_dir}) — cache ကို ဖျက်ပြီး ပြန်စမ်းပါ: {work}"
        )
    return images_dir, labels_dir


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def _parse_label(path: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                except Exception:
                    continue
                rows.append((cid, " ".join(parts[1:])))
    except Exception:
        pass
    return rows


def _select_images(
    labels_dir: str,
    images_dir: str,
    id_map: Dict[int, int],
    per_class: int,
    on_log: Callable[[str], None],
) -> Tuple[List[Tuple[str, str, List[Tuple[int, str]]]], Dict[int, int]]:
    """
    Class တစ်ခုချင်းစီအတွက် အနည်းဆုံး `per_class` instance ရအောင် ပုံများကို
    greedy ရွေးပေးမည်။ ပုံတစ်ပုံမှာ class များစွာပါတတ်လို့ ရွေးလိုက်တာနဲ့ အားလုံး
    တစ်ပြိုင်တည်း တိုးသွားမည်။
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    stem_to_img: Dict[str, str] = {}
    for f in os.listdir(images_dir):
        stem, ext = os.path.splitext(f)
        if ext.lower() in exts:
            stem_to_img[stem] = os.path.join(images_dir, f)

    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
    on_log(f"  📂 label {len(label_files)} ဖိုင် / image {len(stem_to_img)} ပုံ တွေ့ရှိ")

    rng = random.Random(0)   # deterministic
    label_files.sort()
    rng.shuffle(label_files)

    counts: Dict[int, int] = defaultdict(int)
    chosen: List[Tuple[str, str, List[Tuple[int, str]]]] = []
    for lf in label_files:
        stem = os.path.splitext(lf)[0]
        img = stem_to_img.get(stem)
        if not img:
            continue
        rows = _parse_label(os.path.join(labels_dir, lf))
        mapped = [(id_map[c], r) for c, r in rows if c in id_map]
        if not mapped:
            continue
        if any(counts[m] < per_class for m, _ in mapped):
            chosen.append((img, stem, mapped))
            for m, _ in mapped:
                counts[m] += 1
    return chosen, dict(counts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def remove_existing_replay(master_root: str = MASTER_DIR) -> int:
    """အရင်ထည့်ထားသော replay ဖိုင်များ (prefix နဲ့) ကို ဖျက်ပေးမည်။"""
    removed = 0
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            d = os.path.join(master_root, sub, split)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.startswith(REPLAY_PREFIX):
                    try:
                        os.remove(os.path.join(d, f))
                        removed += 1
                    except Exception:
                        pass
    marker = os.path.join(master_root, REPLAY_MARKER)
    if os.path.isfile(marker):
        try:
            os.remove(marker)
        except Exception:
            pass
    return removed


def add_coco_replay(
    master_root: str = MASTER_DIR,
    source: str = "val2017",
    per_class: int = 30,
    val_ratio: float = 0.2,
    replace: bool = True,
    on_log: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    COCO ပုံအချို့ကို master dataset ထဲ ထည့်ပေးမည် — မူလ class ၈၀ မပျောက်စေရန်။

    Args:
        master_root: Master dataset root
        source:      "coco128" (မြန်၊ စမ်းရန်) သို့ "val2017" (တကယ်သုံးရန်)
        per_class:   class တစ်ခုလျှင် ရည်မှန်းထားသော instance အရေအတွက်
        val_ratio:   ရွေးထားသော ပုံများထဲမှ val အဖြစ်ထားမည့် အချိုး
        replace:     True ဆိုရင် အရင်ထည့်ထားသော replay ကို အရင်ဖျက်မည်
    """
    log = on_log or _noop
    if source not in SOURCES:
        return {"ok": False, "message": f"source မမှန်ပါ: {source} (ရနိုင်သည်: {list(SOURCES)})"}
    if per_class < 1:
        return {"ok": False, "message": "per_class သည် ၁ ထက် ကြီးရပါမည်"}

    started = time.time()
    _ensure_master_structure(master_root)
    master_yaml = os.path.join(master_root, "data.yaml")

    # ၁။ Master ရဲ့ class နာမည် → ID mapping (index မဟုတ်ဘဲ နာမည်နဲ့ တွဲသည်)
    master_names = _normalize_names(_load_yaml(master_yaml).get("names", {}))
    name_to_id = {v.strip().lower(): k for k, v in master_names.items()}
    id_map: Dict[int, int] = {}
    missing: List[str] = []
    for coco_id, coco_name in enumerate(COCO_NAMES):
        mid = name_to_id.get(coco_name.lower())
        if mid is None:
            missing.append(coco_name)
        else:
            id_map[coco_id] = mid
    if not id_map:
        return {"ok": False, "message": "Master data.yaml ထဲမှာ COCO class နာမည်တွေ မတွေ့ရှိပါ။"}
    log(f"[Replay] COCO class {len(id_map)}/{len(COCO_NAMES)} ကို master ID နဲ့ တွဲပြီးပါပြီ။")
    if missing:
        log(f"[Replay] ⚠️  master ထဲ မပါသော COCO class {len(missing)} ခုကို ကျော်မည်: {', '.join(missing[:6])}...")

    # ၂။ Download + extract
    spec = SOURCES[source]
    log(f"[Replay] Source: {spec['label']}")
    try:
        images_dir, labels_dir = _prepare_source(source, log)
    except Exception as e:
        return {"ok": False, "message": f"Download/extract မအောင်မြင်ပါ: {e}"}

    # ၃။ ပုံရွေး
    log(f"[Replay] class တစ်ခုလျှင် instance {per_class} ခု ရည်မှန်းပြီး ပုံရွေးနေသည်...")
    chosen, counts = _select_images(labels_dir, images_dir, id_map, per_class, log)
    if not chosen:
        return {"ok": False, "message": "ရွေးစရာ ပုံမတွေ့ရှိပါ (labels နဲ့ images stem မကိုက်ညီပါ)။"}
    log(f"[Replay] ပုံ {len(chosen)} ပုံ ရွေးပြီးပါပြီ။")

    # ၄။ အရင် replay ဖျက်
    if replace:
        n = remove_existing_replay(master_root)
        if n:
            log(f"[Replay] အရင်ထည့်ထားသော replay ဖိုင် {n} ခု ဖျက်ပြီးပါပြီ။")

    # ၅။ master ထဲ ကူး
    rng = random.Random(1)
    n_val = max(1, int(len(chosen) * max(0.0, min(0.9, val_ratio))))
    idxs = list(range(len(chosen)))
    rng.shuffle(idxs)
    val_set = set(idxs[:n_val])

    copied = {"train": 0, "val": 0}
    rows_written = 0
    for i, (img_path, stem, mapped) in enumerate(chosen):
        split = "val" if i in val_set else "train"
        ext = os.path.splitext(img_path)[1]
        new_stem = f"{REPLAY_PREFIX}{stem}"
        dst_img = os.path.join(master_root, "images", split, new_stem + ext)
        dst_lbl = os.path.join(master_root, "labels", split, new_stem + ".txt")
        os.makedirs(os.path.dirname(dst_img), exist_ok=True)
        os.makedirs(os.path.dirname(dst_lbl), exist_ok=True)
        try:
            shutil.copy2(img_path, dst_img)
        except Exception:
            continue
        with open(dst_lbl, "w", encoding="utf-8") as f:
            for mid, rest in mapped:
                f.write(f"{mid} {rest}\n")
                rows_written += 1
        copied[split] += 1

    # ၆။ မှတ်တမ်း
    marker = {
        "source": source,
        "per_class": per_class,
        "images_train": copied["train"],
        "images_val": copied["val"],
        "label_rows": rows_written,
        "added_at": time.time(),
    }
    try:
        with open(os.path.join(master_root, REPLAY_MARKER), "w", encoding="utf-8") as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    id_to_name = {v: k for k, v in name_to_id.items()}
    weakest = sorted(((counts.get(mid, 0), id_to_name.get(mid, str(mid)))
                      for mid in id_map.values()))[:5]
    elapsed = round(time.time() - started, 1)
    msg = (
        f"✅ COCO replay ထည့်ပြီးပါပြီ — ပုံ {copied['train']} (train) + {copied['val']} (val)၊ "
        f"label {rows_written} rows၊ {elapsed}s ကြာသည်။"
    )
    log(f"[Replay] {msg}")
    log(f"[Replay] instance အနည်းဆုံး class များ: " +
        ", ".join(f"{n}={c}" for c, n in weakest))

    return {
        "ok": True,
        "message": msg,
        "source": source,
        "per_class": per_class,
        "images_train": copied["train"],
        "images_val": copied["val"],
        "label_rows": rows_written,
        "classes_covered": len([c for c in counts.values() if c > 0]),
        "weakest_classes": [{"name": n, "count": c} for c, n in weakest],
        "elapsed_sec": elapsed,
    }


def replay_info(master_root: str = MASTER_DIR) -> Dict[str, Any]:
    """Master ထဲ replay ထည့်ထားပြီးလား စစ်ပေးမည်။"""
    marker = os.path.join(master_root, REPLAY_MARKER)
    if not os.path.isfile(marker):
        return {"present": False, "sources": {k: v["label"] for k, v in SOURCES.items()}}
    try:
        with open(marker, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["present"] = True
        data["sources"] = {k: v["label"] for k, v in SOURCES.items()}
        return data
    except Exception as e:
        return {"present": False, "error": str(e),
                "sources": {k: v["label"] for k, v in SOURCES.items()}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:
    import argparse
    # Windows console (cp1252) မှာ မြန်မာစာ print လုပ်လို့ crash မဖြစ်စေရန်
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(description="COCO replay ကို master dataset ထဲ ထည့်ရန်")
    p.add_argument("--master", default=MASTER_DIR, help="Master dataset root")
    p.add_argument("--source", default="val2017", choices=list(SOURCES),
                   help="coco128 = မြန် (၇MB) | val2017 = ကောင်း (၈၀၀MB)")
    p.add_argument("--per-class", type=int, default=30, help="class တစ်ခုလျှင် instance အရေအတွက်")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--keep-existing", action="store_true",
                   help="အရင်ထည့်ထားသော replay ကို မဖျက်ဘဲ ထပ်ထည့်မည်")
    p.add_argument("--remove", action="store_true", help="replay ဖိုင်များကို ဖျက်ရုံသာ လုပ်မည်")
    args = p.parse_args()

    if args.remove:
        n = remove_existing_replay(args.master)
        print(f"replay ဖိုင် {n} ခု ဖျက်ပြီးပါပြီ။")
        return

    res = add_coco_replay(
        master_root=args.master,
        source=args.source,
        per_class=args.per_class,
        val_ratio=args.val_ratio,
        replace=not args.keep_existing,
        on_log=lambda s: print(s, flush=True),
    )
    print(json.dumps({k: v for k, v in res.items() if k != "weakest_classes"},
                     ensure_ascii=False, indent=2))
    if not res.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    _cli()
