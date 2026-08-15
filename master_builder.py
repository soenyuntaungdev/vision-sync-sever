"""
VisionSync ΓÇö Master Dataset Builder & Continuous Fine-Tuning Helper

ßÇÖßÇ░ßÇ£ ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇößÇ╛ßÇäßÇ╖ßÇ║ ßÇòßÇàßÇ╣ßÇàßÇèßÇ║ßÇ╕ßÇíßÇ₧ßÇàßÇ║ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÉßÇèßÇ║ßÇ╕ßÇ₧ßÇ▒ßÇ¼ .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ ßÇíßÇûßÇ╝ßÇàßÇ║
ßÇíßÇ£ßÇ¡ßÇ»ßÇíßÇ£ßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ ßÇòßÇ▒ßÇ½ßÇäßÇ║ßÇ╕ßÇàßÇòßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ╖ßÇ║ Tools ßÇÖßÇ╗ßÇ¼ßÇ╕

Usage (CLI):
    # ßüüßüï master dataset ßÇ¢ßÇ▓ßÇ╖ data.yaml ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ class ßÇíßÇ₧ßÇàßÇ║ ßÇæßÇèßÇ╖ßÇ║ßÇüßÇ╝ßÇäßÇ║ßÇ╕
    python master_builder.py add-class --yaml dataset/master/data.yaml --name my_burmese_coin

    # ßüéßüï ßÇòßÇàßÇ╣ßÇàßÇèßÇ║ßÇ╕ßÇíßÇ₧ßÇàßÇ║ (ßÇÑßÇòßÇÖßÇ¼- Roboflow ßÇÖßÇ╛ export ßÇ£ßÇ»ßÇòßÇ║ßÇæßÇ¼ßÇ╕ßÇ₧ßÇ▒ßÇ¼ dataset) ßÇÇßÇ¡ßÇ»
    #    master dataset ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ label ID auto shift ßÇòßÇ╝ßÇ»ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ merge ßÇ£ßÇ»ßÇòßÇ║ßÇüßÇ╝ßÇäßÇ║ßÇ╕
    python master_builder.py merge-dataset ^
        --master dataset/master ^
        --source dataset/Traffic_Light_v3i_yolov8 ^
        --class-name traffic_light_myanmar

    # ßüâßüï Base .pt (ßÇÖßÇ░ßÇ£ ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕) + master dataset ßÇíßÇ₧ßÇàßÇ║ ßÇûßÇ╝ßÇäßÇ╖ßÇ║ continuous fine-tune
    python master_builder.py finetune ^
        --base models/my_old_80_classes.pt ^
        --yaml dataset/master/data.yaml ^
        --epochs 20 ^
        --name visionsync_master

Usage (Python API):
    from master_builder import add_class_to_yaml, merge_dataset_into_master, continuous_finetune
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from dataset_utils import fixup_dataset, preflight_check_training

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(BACKEND_DIR, "dataset", "master")
RUNS_DIR = os.path.join(BACKEND_DIR, "runs", "detect")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")

COCO_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("Γ¥î PyYAML ßÇÖßÇÉßÇòßÇ║ßÇåßÇäßÇ║ßÇ¢ßÇ₧ßÇ▒ßÇ╕ßÇòßÇ½ßüï pip install pyyaml ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇÉßÇòßÇ║ßÇåßÇäßÇ║ßÇòßÇ½ßüï")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("Γ¥î PyYAML ßÇÖßÇÉßÇòßÇ║ßÇåßÇäßÇ║ßÇ¢ßÇ₧ßÇ▒ßÇ╕ßÇòßÇ½ßüï pip install pyyaml ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇÉßÇòßÇ║ßÇåßÇäßÇ║ßÇòßÇ½ßüï")
        sys.exit(1)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False, allow_unicode=True, width=120)


def _ensure_master_structure(master_root: str) -> None:
    for sub in (
        os.path.join("images", "train"),
        os.path.join("images", "val"),
        os.path.join("labels", "train"),
        os.path.join("labels", "val"),
    ):
        os.makedirs(os.path.join(master_root, sub), exist_ok=True)
    yaml_path = os.path.join(master_root, "data.yaml")
    if not os.path.isfile(yaml_path):
        data: Dict[str, Any] = {
            "path": os.path.relpath(master_root, BACKEND_DIR).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "nc": len(COCO_NAMES),
            "names": {i: name for i, name in enumerate(COCO_NAMES)},
        }
        _save_yaml(yaml_path, data)
        print(f"Γ£à Master data.yaml ßÇíßÇ₧ßÇàßÇ║ ßÇûßÇößÇ║ßÇÉßÇ«ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï COCO {len(COCO_NAMES)} classes")
    # Auto-fixup: nested folders, absolute paths, nc/names sync
    try:
        fixup_dataset(master_root, verbose=False)
    except Exception:
        pass


def _normalize_names(names: Any) -> Dict[int, str]:
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        out: Dict[int, str] = {}
        for k, v in names.items():
            try:
                out[int(k)] = str(v)
            except Exception:
                pass
        return out
    return {}


# ---------------------------------------------------------------------------
# Public API 1 ΓÇö data.yaml ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ class ßÇíßÇ₧ßÇàßÇ║ auto ßÇæßÇèßÇ╖ßÇ║ßÇòßÇ▒ßÇ╕ßÇ¢ßÇößÇ║
# ---------------------------------------------------------------------------
def add_class_to_yaml(
    yaml_path: str,
    new_class_name: str,
) -> Dict[str, Any]:
    """
    ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ data.yaml ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ class ßÇößÇ¼ßÇÖßÇèßÇ║ßÇíßÇ₧ßÇàßÇ║ßÇÇßÇ¡ßÇ» ßÇößÇ▒ßÇ¼ßÇÇßÇ║ßÇåßÇ»ßÇ╢ßÇ╕ ID ßÇößÇ╛ßÇäßÇ╖ßÇ║ auto ßÇæßÇèßÇ╖ßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï
    ßÇæßÇ¡ßÇ»ßÇößÇ¼ßÇÖßÇèßÇ║ ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇÇ ID ßÇ₧ßÇ¼ return ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ ßÇòßÇ╝ßÇößÇ║ßÇæßÇèßÇ╖ßÇ║ßÇÖßÇèßÇ║ßÇÖßÇƒßÇ»ßÇÉßÇ║ßÇòßÇ½ßüï

    Returns:
        { "ok": bool, "class_id": int, "total_nc": int, "already_exists": bool, "message": str }
    """
    if not os.path.isfile(yaml_path):
        return {"ok": False, "class_id": -1, "total_nc": -1, "already_exists": False,
                "message": f"data.yaml ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {yaml_path}"}

    data = _load_yaml(yaml_path)
    names = _normalize_names(data.get("names", {}))
    name_clean = new_class_name.strip()

    for cid, cname in names.items():
        if cname.lower() == name_clean.lower():
            nc = int(data.get("nc", max(names.keys()) + 1 if names else 0))
            return {
                "ok": True,
                "class_id": cid,
                "total_nc": nc,
                "already_exists": True,
                "message": f"'{name_clean}' ßÇ₧ßÇèßÇ║ ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ Class ID {cid} ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï",
            }

    next_id = max(names.keys()) + 1 if names else 0
    names[next_id] = name_clean
    data["names"] = dict(sorted(names.items()))
    data["nc"] = len(names)
    _save_yaml(yaml_path, data)

    return {
        "ok": True,
        "class_id": next_id,
        "total_nc": data["nc"],
        "already_exists": False,
        "message": f"ßÇíßÇåßÇäßÇ║ßÇòßÇ╝ßÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï '{name_clean}' ßÇÇßÇ¡ßÇ» Class ID {next_id} ßÇíßÇûßÇ╝ßÇàßÇ║ ßÇæßÇèßÇ╖ßÇ║ßÇ₧ßÇ╜ßÇäßÇ║ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï (Total: {data['nc']})",
    }


# ---------------------------------------------------------------------------
# Public API 2 ΓÇö Source dataset ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÇßÇ¡ßÇ» master dataset ßÇæßÇ▓ label shift ßÇòßÇ╝ßÇ»ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ merge
# ---------------------------------------------------------------------------
def _remap_label_file(label_path: str, out_path: str, id_map: Dict[int, int]) -> Tuple[int, int]:
    """
    Source label file ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÇßÇ¡ßÇ» master ßÇ¢ßÇ▓ßÇ╖ class ID ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇíßÇûßÇ╝ßÇàßÇ║ ßÇòßÇ╝ßÇößÇ║ map ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    id_map ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼ ßÇÖßÇòßÇ½ßÇ₧ßÇ▒ßÇ¼ source class ID ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» ßÇüßÇ╗ßÇößÇ║ßÇæßÇ¼ßÇ╕ßÇüßÇ▓ßÇ╖ßÇ₧ßÇèßÇ║ (dropped)ßüï
    ßÇÜßÇüßÇäßÇ║ code ßÇÇ `cid + offset` ßÇûßÇ╝ßÇäßÇ╖ßÇ║ shift ßÇ£ßÇ»ßÇòßÇ║ßÇüßÇ▓ßÇ╖ßÇ¢ßÇ¼ßüè source ßÇÖßÇ╛ßÇ¼ class ßÇÉßÇàßÇ║ßÇüßÇ»ßÇæßÇÇßÇ║
    ßÇòßÇ¡ßÇ»ßÇòßÇ½ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ master ßÇ¢ßÇ▓ßÇ╖ nc ßÇÇßÇ¡ßÇ» ßÇÇßÇ╗ßÇ▒ßÇ¼ßÇ║ßÇ£ßÇ╜ßÇößÇ║ßÇ₧ßÇ▒ßÇ¼ ID ßÇÖßÇ╗ßÇ¼ßÇ╕ ßÇæßÇ╜ßÇÇßÇ║ßÇ£ßÇ¼ßÇòßÇ╝ßÇ«ßÇ╕ dataset ßÇòßÇ╗ßÇÇßÇ║ßÇüßÇ▓ßÇ╖ßÇ₧ßÇèßÇ║ßüï
    """
    copied = 0
    dropped = 0
    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    new_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            cid = int(parts[0])
        except Exception:
            continue
        if cid not in id_map:
            dropped += 1
            continue
        parts[0] = str(id_map[cid])
        new_lines.append(" ".join(parts) + "\n")
        copied += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    return copied, dropped


def merge_dataset_into_master(
    master_root: str,
    source_root: str,
    new_class_name: str,
    source_class_ids: Optional[List[int]] = None,
    merge_mode: str = "auto",
) -> Dict[str, Any]:
    """
    Roboflow export ßÇ£ßÇ»ßÇòßÇ║ßÇæßÇ¼ßÇ╕ßÇ₧ßÇ▒ßÇ¼ (ßÇ₧ßÇ¡ßÇ»ßÇ╖) ßÇÉßÇàßÇ║ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ßÇüßÇ╗ßÇäßÇ║ßÇ╕ßÇ¢ßÇ¢ßÇ╛ßÇ¡ßÇ₧ßÇèßÇ╖ßÇ║ dataset ßÇíßÇ₧ßÇàßÇ║ßÇÇßÇ¡ßÇ»
    master dataset ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ ßÇíßÇ▒ßÇ¼ßÇÇßÇ║ßÇòßÇ½ßÇíßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ merge ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    ßüüßüï source ßÇ¢ßÇ▓ßÇ╖ class ßÇÉßÇàßÇ║ßÇüßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ╕ßÇàßÇ«ßÇíßÇÉßÇ╜ßÇÇßÇ║ master ßÇæßÇ▓ßÇÇ ID ßÇÇßÇ¡ßÇ» ßÇåßÇ»ßÇ╢ßÇ╕ßÇûßÇ╝ßÇÉßÇ║ (mapping ßÇåßÇ▒ßÇ¼ßÇÇßÇ║)
    ßüéßüï ßÇ£ßÇ¡ßÇ»ßÇíßÇòßÇ║ßÇ₧ßÇ▒ßÇ¼ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» master ßÇ¢ßÇ▓ßÇ╖ data.yaml ßÇæßÇ▓ ßÇæßÇèßÇ╖ßÇ║ (ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║ reuse)
    ßüâßüï source ßÇ¢ßÇ▓ßÇ╖ label file (train/val) ßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ßÇÇßÇ¡ßÇ» mapping ßÇíßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ ßÇòßÇ╝ßÇößÇ║ßÇ¢ßÇ▒ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ master labels ßÇæßÇ▓ ßÇÇßÇ░ßÇ╕
    ßüäßüï image files ßÇÉßÇ╜ßÇ▒ßÇÇßÇ¡ßÇ»ßÇ£ßÇèßÇ║ßÇ╕ master images ßÇæßÇ▓ ßÇÇßÇ░ßÇ╕ (prefix ßÇÇßÇ¡ßÇ» unique ID ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇòßÇ░ßÇ╕ßÇÉßÇ╜ßÇ▓)

    Args:
        master_root:    Master dataset root (data.yaml ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ folder)
        source_root:    Source dataset root (ßÇòßÇ»ßÇ╢ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇößÇ╛ßÇäßÇ╖ßÇ║ labels ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ folder)
        new_class_name: Master ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ßÇæßÇèßÇ╖ßÇ║ßÇÖßÇèßÇ╖ßÇ║ (ßÇ₧ßÇ¡ßÇ»ßÇ╖) ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ßÇòßÇ╝ßÇößÇ║ßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇÖßÇèßÇ╖ßÇ║ class ßÇößÇ¼ßÇÖßÇèßÇ║
        source_class_ids: Source ßÇæßÇ▓ßÇÖßÇ╛ ßÇñßÇíßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ßÇíßÇàßÇ¼ßÇ╕ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ»ßÇ₧ßÇ¼ ßÇÜßÇ░ßÇÖßÇèßÇ║ (None = ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕)
        merge_mode:
            "collapse"  ΓÇö source ßÇ¢ßÇ▓ßÇ╖ class ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÇßÇ¡ßÇ» new_class_name ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÉßÇèßÇ║ßÇ╕ßÇíßÇûßÇ╝ßÇàßÇ║ ßÇòßÇ▒ßÇ½ßÇäßÇ║ßÇ╕
            "per_class" ΓÇö source ßÇ¢ßÇ▓ßÇ╖ class ßÇößÇ¼ßÇÖßÇèßÇ║ßÇÉßÇàßÇ║ßÇüßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ╕ßÇàßÇ«ßÇÇßÇ¡ßÇ» master ßÇæßÇ▓ ßÇ₧ßÇ«ßÇ╕ßÇ₧ßÇößÇ╖ßÇ║ßÇæßÇèßÇ╖ßÇ║
            "auto"      ΓÇö (default) class ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÉßÇèßÇ║ßÇ╕ ßÇÆßÇ½ßÇÖßÇ╛ßÇÖßÇƒßÇ»ßÇÉßÇ║ source_class_ids
                          ßÇ¢ßÇ╜ßÇ▒ßÇ╕ßÇæßÇ¼ßÇ╕ßÇ¢ßÇäßÇ║ collapseßüè ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇößÇ▒ßÇ¢ßÇäßÇ║ per_class

    Returns:
        Summary dict with counts
    """
    _ensure_master_structure(master_root)
    master_yaml = os.path.join(master_root, "data.yaml")
    source_yaml = os.path.join(source_root, "data.yaml")

    # Auto-fixup source dataset (Roboflow export ΓåÆ standard layout, yaml normalize)
    try:
        fixup_dataset(source_root, verbose=True)
    except Exception:
        pass
    source_yaml = os.path.join(source_root, "data.yaml")

    if not os.path.isfile(source_yaml):
        return {"ok": False, "message": f"Source data.yaml ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {source_yaml}"}

    # ßüüßüï Source yaml ßÇûßÇÉßÇ║
    src_data = _load_yaml(source_yaml)
    src_names = _normalize_names(src_data.get("names", {}))
    src_nc = int(src_data.get("nc", (max(src_names.keys()) + 1) if src_names else 0))

    # ßüéßüï Source class ΓåÆ Master class ID mapping ßÇÉßÇèßÇ║ßÇåßÇ▒ßÇ¼ßÇÇßÇ║
    #    (ßÇÜßÇüßÇäßÇ║ßÇÇ `cid + offset` shift ßÇ£ßÇ»ßÇòßÇ║ßÇüßÇ▓ßÇ╖ßÇ£ßÇ¡ßÇ»ßÇ╖ source ßÇÖßÇ╛ßÇ¼ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇößÇ▒ßÇ¢ßÇäßÇ║
    #     master ßÇ¢ßÇ▓ßÇ╖ nc ßÇÇßÇ¡ßÇ»ßÇÇßÇ╗ßÇ▒ßÇ¼ßÇ║ßÇÉßÇ▓ßÇ╖ label ID ßÇÉßÇ╜ßÇ▒ ßÇæßÇ╜ßÇÇßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ dataset ßÇòßÇ╗ßÇÇßÇ║ßÇüßÇ▓ßÇ╖ßÇ₧ßÇèßÇ║ßüï)
    candidate_ids: List[int] = sorted(src_names.keys()) if src_names else list(range(max(0, src_nc)))
    if source_class_ids:
        wanted = set(source_class_ids)
        candidate_ids = [i for i in candidate_ids if i in wanted]
    if not candidate_ids:
        return {"ok": False, "message": "Source dataset ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼ ßÇÜßÇ░ßÇàßÇ¢ßÇ¼ class ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½ßüï"}

    mode = (merge_mode or "auto").lower()
    if mode not in ("auto", "collapse", "per_class"):
        return {"ok": False, "message": f"merge_mode ßÇÖßÇÖßÇ╛ßÇößÇ║ßÇòßÇ½: {merge_mode}"}
    if mode == "auto":
        mode = "collapse" if (len(candidate_ids) <= 1 or source_class_ids) else "per_class"

    id_map: Dict[int, int] = {}
    added_classes: List[Dict[str, Any]] = []
    if mode == "collapse":
        add_result = add_class_to_yaml(master_yaml, new_class_name)
        if not add_result["ok"]:
            return {"ok": False, "message": add_result["message"]}
        for i in candidate_ids:
            id_map[i] = add_result["class_id"]
        added_classes.append({"name": new_class_name, "id": add_result["class_id"],
                              "already_exists": add_result["already_exists"]})
    else:
        for i in candidate_ids:
            nm = (src_names.get(i) or "").strip() or f"{new_class_name}_{i}"
            ar = add_class_to_yaml(master_yaml, nm)
            if not ar["ok"]:
                return {"ok": False, "message": ar["message"]}
            id_map[i] = ar["class_id"]
            added_classes.append({"name": nm, "id": ar["class_id"],
                                  "already_exists": ar["already_exists"]})
        add_result = {"class_id": added_classes[0]["id"],
                      "total_nc": len(_normalize_names(_load_yaml(master_yaml).get("names", {}))),
                      "already_exists": all(a["already_exists"] for a in added_classes)}
    master_class_id = add_result["class_id"]

    # ßüâßüï Source ßÇ¢ßÇ▓ßÇ╖ train/val folder ßÇÉßÇ╜ßÇ▒ ßÇ¢ßÇ╛ßÇ¼
    # Roboflow (A) ΓÇö {split}/images/ pattern OR YOLOv8 (B) ΓÇö images/{split}/ pattern ßÇößÇ╛ßÇàßÇ║ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÇßÇ¡ßÇ» ßÇàßÇàßÇ║ßÇÖßÇÜßÇ║
    splits: List[Tuple[str, str, str, str]] = []  # (split_name, target_split, img_dir_abs, lbl_dir_abs)
    split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    for split, tgt in split_aliases.items():
        # Pattern A: Roboflow classic ΓÇö root/train/images/, root/train/labels/
        a_img = os.path.join(source_root, split, "images")
        a_lbl = os.path.join(source_root, split, "labels")
        if os.path.isdir(a_img) and os.path.isdir(a_lbl):
            splits.append((split, tgt, a_img, a_lbl))
            continue
        # Pattern B: Ultralytics/YOLOv8 standard ΓÇö root/images/train/, root/labels/train/
        b_img = os.path.join(source_root, "images", split)
        b_lbl = os.path.join(source_root, "labels", split)
        if os.path.isdir(b_img) and os.path.isdir(b_lbl):
            splits.append((split, tgt, b_img, b_lbl))
            continue

    if not splits:
        # Debug: ßÇÿßÇÜßÇ║ folders ßÇÉßÇ╜ßÇ▒ßÇ¢ßÇ╛ßÇ¡ßÇ£ßÇ▓ ßÇàßÇàßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ error message ßÇæßÇ▓ßÇæßÇèßÇ╖ßÇ║ßÇòßÇ▒ßÇ╕ßÇÖßÇÜßÇ║
        debug_items = []
        if os.path.isdir(source_root):
            for name in sorted(os.listdir(source_root)):
                fp = os.path.join(source_root, name)
                if os.path.isdir(fp):
                    try:
                        n_child = len(os.listdir(fp))
                    except Exception:
                        n_child = -1
                    debug_items.append(f"{name}/({n_child})")
                else:
                    debug_items.append(name)
        hint = ""
        if debug_items:
            hint = " Root contents: " + ", ".join(debug_items[:12])
        return {"ok": False, "message": f"Source ßÇæßÇ▓ßÇÉßÇ╜ßÇäßÇ║ train/val images/labels folders ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½ßüï{hint}"}

    master_data = _load_yaml(master_yaml)
    master_rel_root = master_data.get("path") or "."
    master_abs = os.path.normpath(os.path.join(os.path.dirname(master_yaml), master_rel_root)) \
        if not os.path.isabs(master_rel_root) else master_rel_root

    import uuid
    run_tag = uuid.uuid4().hex[:6]
    total_images = 0
    total_labels = 0
    total_dropped = 0

    for src_split, dst_split, src_img_dir, src_lbl_dir in splits:
        dst_img_dir = os.path.join(master_abs, "images", dst_split)
        dst_lbl_dir = os.path.join(master_abs, "labels", dst_split)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)

        for fname in sorted(os.listdir(src_img_dir)):
            src_img = os.path.join(src_img_dir, fname)
            if not os.path.isfile(src_img):
                continue
            stem, ext = os.path.splitext(fname)
            new_fname = f"{run_tag}_{stem}{ext}"
            dst_img = os.path.join(dst_img_dir, new_fname)
            shutil.copy2(src_img, dst_img)
            total_images += 1

            src_lbl = os.path.join(src_lbl_dir, stem + ".txt")
            dst_lbl = os.path.join(dst_lbl_dir, f"{run_tag}_{stem}.txt")
            if os.path.isfile(src_lbl):
                c, d = _remap_label_file(src_lbl, dst_lbl, id_map)
                total_labels += c
                total_dropped += d
            else:
                open(dst_lbl, "w", encoding="utf-8").close()

    final_nc = len(_normalize_names(_load_yaml(master_yaml).get("names", {})))
    if mode == "collapse":
        cls_desc = f"Class '{new_class_name}' (ID {master_class_id})"
    else:
        cls_desc = "Class " + ", ".join(f"'{a['name']}'(ID {a['id']})" for a in added_classes)

    return {
        "ok": True,
        "class_id": master_class_id,
        "class_name": new_class_name,
        "merge_mode": mode,
        "classes": added_classes,
        "class_map": {str(k): v for k, v in sorted(id_map.items())},
        "master_nc": final_nc,
        "already_exists": add_result["already_exists"],
        "merged_images": total_images,
        "merged_labels": total_labels,
        "dropped_label_rows": total_dropped,
        "message": (
            f"ßÇíßÇåßÇäßÇ║ßÇòßÇ╝ßÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï {cls_desc} ßÇíßÇ¼ßÇ╕ "
            f"images {total_images} ßÇüßÇ» + labels {total_labels} rows ßÇÇßÇ¡ßÇ» master dataset ßÇæßÇ▓ merge ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï "
            f"(mode={mode}, total nc={final_nc})"
        ),
    }


# ---------------------------------------------------------------------------
# Public API 3 ΓÇö Continuous fine-tune
# ---------------------------------------------------------------------------
class _LogForwarder(io.StringIO):
    """Captures prints AND forwards them to an optional sink (e.g. API state)."""

    def __init__(self, sink: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._sink = sink
        # sys.stdout ßÇÖßÇƒßÇ»ßÇÉßÇ║ßÇÿßÇ▓ sys.__stdout__ ßÇÇßÇ¡ßÇ» ßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇ₧ßÇèßÇ║ ΓÇö capture ßÇÉßÇàßÇ║ßÇüßÇ»ßÇòßÇ▒ßÇ½ßÇ║ ßÇößÇ▒ßÇ¼ßÇÇßÇ║ßÇÉßÇàßÇ║ßÇüßÇ»
        # ßÇæßÇòßÇ║ßÇÉßÇäßÇ║ßÇÖßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ log ßÇÉßÇ╜ßÇ▒ ßÇößÇ╛ßÇàßÇ║ßÇüßÇ½ßÇæßÇòßÇ║ßÇæßÇ╜ßÇÇßÇ║ßÇÉßÇ¼ (ßÇ₧ßÇ¡ßÇ»ßÇ╖) loop ßÇûßÇ╝ßÇàßÇ║ßÇÉßÇ¼ ßÇÖßÇûßÇ╝ßÇàßÇ║ßÇàßÇ▒ßÇ¢ßÇößÇ║ßüï
        self._real = sys.__stdout__

    def write(self, s: str) -> int:
        if self._sink:
            stripped = s.rstrip("\n")
            if stripped:
                try:
                    self._sink(stripped)
                except Exception:
                    pass
        try:
            self._real.write(s)
            self._real.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            self._real.flush()
        except Exception:
            pass


def continuous_finetune(
    base_model_path: str,
    data_yaml_path: str,
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.001,
    run_name: str = "visionsync_master",
    freeze: int = 10,
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    """
    ßÇÖßÇ░ßÇ£ ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ base .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇÇßÇ¡ßÇ» ßÇÖßÇ░ßÇÉßÇèßÇ║ßÇòßÇ╝ßÇ«ßÇ╕
    class ßÇíßÇ₧ßÇàßÇ║ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇòßÇ½ßÇ₧ßÇ▒ßÇ¼ master dataset ßÇûßÇ╝ßÇäßÇ╖ßÇ║ continuous fine-tune ßÇòßÇ╝ßÇ»ßÇ£ßÇ»ßÇòßÇ║ßÇ₧ßÇèßÇ║ßüï

    - epochs 15-25 ßüè lr0=0.001 ßÇÇ Continuous ßÇíßÇÉßÇ╜ßÇÇßÇ║ standard ßÇòßÇ½ßüï
    - freeze=10 ΓåÆ backbone ßüüßüÇ layer ßÇÇßÇ¡ßÇ» ßÇüßÇ▓ßÇæßÇ¼ßÇ╕ßÇ₧ßÇèßÇ║ßüï ßÇÆßÇ½ßÇÖßÇ£ßÇ»ßÇòßÇ║ßÇ¢ßÇäßÇ║ ßÇòßÇàßÇ╣ßÇàßÇèßÇ║ßÇ╕ßÇíßÇ₧ßÇàßÇ║
      ßÇÉßÇàßÇ║ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ßÇÉßÇèßÇ║ßÇ╕ßÇ₧ßÇ¼ßÇòßÇ½ßÇ₧ßÇ▒ßÇ¼ dataset ßÇößÇ▓ßÇ╖ train ßÇ£ßÇ¡ßÇ»ßÇÇßÇ║ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ model ßÇÇ ßÇÖßÇ░ßÇ£ COCO ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ßÇÇßÇ¡ßÇ»
      ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇ¥ ßÇÖßÇ▒ßÇ╖ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ (catastrophic forgetting) ßÇ¢ßÇ£ßÇ¼ßÇ₧ßÇ▒ßÇ¼ .pt ßÇÇ ßÇÖßÇ░ßÇ£ yolov8n.pt ßÇæßÇÇßÇ║
      ßÇåßÇ¡ßÇ»ßÇ╕ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇÉßÇÉßÇ║ßÇ₧ßÇèßÇ║ßüï
    - ßÇ¢ßÇ£ßÇÆßÇ║ best.pt ßÇÇßÇ¡ßÇ» models/ folder ßÇæßÇ▓ßÇÉßÇ╜ßÇäßÇ║ßÇ£ßÇèßÇ║ßÇ╕ backup ßÇÇßÇ░ßÇ╕ßÇ₧ßÇ¡ßÇÖßÇ║ßÇ╕ßÇ₧ßÇèßÇ║ßüï
    - on_log(line): stdout/stderr ßÇàßÇ¼ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ forward ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï
    - on_progress(epoch, total_epochs): epoch ßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ ßÇüßÇ▒ßÇ½ßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    Returns dict with result info.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"ok": False, "message": "Ultralytics ßÇÖßÇÉßÇòßÇ║ßÇåßÇäßÇ║ßÇ¢ßÇ₧ßÇ▒ßÇ╕ßÇòßÇ½ßüï pip install ultralytics"}

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    cap_out = cap_err = None
    if on_log:
        cap_out = _LogForwarder(on_log)
        cap_err = _LogForwarder(on_log)
        sys.stdout = cap_out
        sys.stderr = cap_err
    try:
        return _continuous_finetune_inner(
            base_model_path=base_model_path,
            data_yaml_path=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            run_name=run_name,
            freeze=freeze,
            on_progress=on_progress,
        )
    finally:
        # ßÇÉßÇüßÇ╝ßÇ¼ßÇ╕ job ßÇÇ ßÇÇßÇ╝ßÇ¼ßÇ╕ßÇûßÇ╝ßÇÉßÇ║ßÇÜßÇ░ßÇæßÇ¼ßÇ╕ßÇ¢ßÇäßÇ║ ßÇÖßÇûßÇ╗ßÇÇßÇ║ßÇÖßÇ¡ßÇàßÇ▒ßÇ¢ßÇößÇ║ ßÇÇßÇ¡ßÇ»ßÇÜßÇ╖ßÇ║ßÇƒßÇ¼ßÇíßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ßÇÖßÇ╛ßÇ₧ßÇ¼ ßÇòßÇ╝ßÇößÇ║ßÇæßÇ¼ßÇ╕ßÇÖßÇèßÇ║
        if cap_out is not None and sys.stdout is cap_out:
            sys.stdout = old_stdout
        if cap_err is not None and sys.stderr is cap_err:
            sys.stderr = old_stderr


def audit_master_coverage(data_yaml_path: str) -> Dict[str, Any]:
    """
    Master dataset ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼ class ßÇÉßÇàßÇ║ßÇüßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ╕ßÇàßÇ«ßÇíßÇÉßÇ╜ßÇÇßÇ║ label ßÇÿßÇÜßÇ║ßÇößÇ╛ßÇàßÇ║ßÇüßÇ»ßÇ¢ßÇ╛ßÇ¡ßÇ£ßÇ▓ ßÇ¢ßÇ▒ßÇÉßÇ╜ßÇÇßÇ║ßÇòßÇ╝ßÇ«ßÇ╕
    "label ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇ¥ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇ₧ßÇ▒ßÇ¼ class" ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» ßÇòßÇ╝ßÇößÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    ßÇñ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇ₧ßÇèßÇ║ fine-tune ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ßÇüßÇ╗ßÇ¡ßÇößÇ║ßÇÖßÇ╛ßÇ¼ ßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇÖßÇèßÇ╖ßÇ║ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇûßÇ╝ßÇàßÇ║ßÇ₧ßÇèßÇ║
    (model ßÇÇ negative ßÇ₧ßÇ¼ßÇÖßÇ╝ßÇäßÇ║ßÇ¢ßÇ₧ßÇûßÇ╝ßÇäßÇ╖ßÇ║)ßüï
    """
    out: Dict[str, Any] = {"ok": False, "counts": {}, "empty_classes": [], "message": ""}
    if not os.path.isfile(data_yaml_path):
        out["message"] = f"data.yaml ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {data_yaml_path}"
        return out
    data = _load_yaml(data_yaml_path)
    names = _normalize_names(data.get("names", {}))
    root = data.get("path") or os.path.dirname(os.path.abspath(data_yaml_path))
    if not os.path.isabs(root):
        root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(data_yaml_path)), root))

    counts: Dict[int, int] = {cid: 0 for cid in names}
    for split in ("train", "val"):
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            try:
                with open(os.path.join(lbl_dir, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            cid = int(parts[0])
                        except Exception:
                            continue
                        counts[cid] = counts.get(cid, 0) + 1
            except Exception:
                continue

    empty = [names[cid] for cid in sorted(names) if counts.get(cid, 0) == 0]
    out["ok"] = True
    out["counts"] = {names.get(cid, str(cid)): n for cid, n in sorted(counts.items())}
    out["empty_classes"] = empty
    out["message"] = (
        f"Class {len(names)} ßÇüßÇ»ßÇíßÇößÇÇßÇ║ {len(empty)} ßÇüßÇ»ßÇÖßÇ╛ßÇ¼ label ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇ¥ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½ßüï"
        if empty else f"Class {len(names)} ßÇüßÇ»ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÖßÇ╛ßÇ¼ label ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï"
    )
    return out


def _continuous_finetune_inner(
    base_model_path: str,
    data_yaml_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    run_name: str,
    freeze: int = 10,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(base_model_path):
        return {"ok": False, "message": f"Base model ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {base_model_path}"}
    if not os.path.isfile(data_yaml_path):
        return {"ok": False, "message": f"data.yaml ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {data_yaml_path}"}

    # Auto-fixup dataset root ßÇößÇ╛ßÇäßÇ╖ßÇ║ preflight validation
    try:
        dataset_root = os.path.dirname(os.path.abspath(data_yaml_path))
        fixup_dataset(dataset_root, verbose=True)
    except Exception:
        pass
    try:
        pf = preflight_check_training(data_yaml_path)
        if not pf.get("ok", False):
            return {
                "ok": False,
                "message": "Dataset preflight ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½ßüï " + " | ".join(pf.get("errors", [])),
                "preflight": pf,
            }
        for w in pf.get("warnings", []) or []:
            print(f"[Warning] {w}")
    except Exception as e:
        return {"ok": False, "message": f"Preflight check error: {e}"}

    # Base model ßÇ¢ßÇ▓ßÇ╖ nc ßÇößÇ╛ßÇäßÇ╖ßÇ║ dataset ßÇ¢ßÇ▓ßÇ╖ nc ßÇÖßÇÉßÇ░ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ Ultralytics ßÇÇ detection head ßÇíßÇ₧ßÇàßÇ║
    # ßÇòßÇ╝ßÇößÇ║ßÇÉßÇèßÇ║ßÇåßÇ▒ßÇ¼ßÇÇßÇ║ßÇ₧ßÇèßÇ║ ΓÇö ßÇÖßÇ░ßÇ£ class weights ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ ßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇ₧ßÇèßÇ║ßüï ßÇÆßÇ½ßÇÇßÇ¡ßÇ» ßÇÇßÇ╝ßÇ¡ßÇ»ßÇ₧ßÇ¡ßÇàßÇ▒ßÇÖßÇèßÇ║ßüï
    try:
        base_info = extract_model_info(base_model_path)
        ds_cfg = _load_yaml(data_yaml_path)
        ds_nc = int(ds_cfg.get("nc", 0) or 0)
        base_nc = int(base_info.get("nc", 0) or 0) if base_info.get("ok") else 0
        if base_nc and ds_nc and base_nc != ds_nc:
            print(f"ΓÜá∩╕Å  Base model nc={base_nc} ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ dataset nc={ds_nc} ΓÇö ßÇÖßÇÉßÇ░ßÇòßÇ½ßüï")
            print("   Ultralytics ßÇÇ detection head ßÇÇßÇ¡ßÇ» ßÇíßÇ₧ßÇàßÇ║ßÇòßÇ╝ßÇößÇ║ßÇÉßÇèßÇ║ßÇåßÇ▒ßÇ¼ßÇÇßÇ║ßÇÖßÇèßÇ║ßÇûßÇ╝ßÇàßÇ║ßüì ßÇÖßÇ░ßÇ£ class")
            print("   ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÇßÇ¡ßÇ» dataset ßÇæßÇ▓ßÇÇ ßÇòßÇ»ßÇ╢ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇòßÇ╝ßÇößÇ║ßÇ₧ßÇäßÇ║ßÇòßÇ▒ßÇ╕ßÇ¢ßÇòßÇ½ßÇÖßÇèßÇ║ßüï Dataset ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼")
            print("   ßÇÖßÇ░ßÇ£ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßüÅ ßÇòßÇ»ßÇ╢ßÇÖßÇ╗ßÇ¼ßÇ╕ ßÇÖßÇòßÇ½ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ ßÇæßÇ¡ßÇ» class ßÇÖßÇ╗ßÇ¼ßÇ╕ ßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇòßÇ½ßÇ£ßÇ¡ßÇÖßÇ╖ßÇ║ßÇÖßÇèßÇ║ßüï")
    except Exception:
        pass

    # Catastrophic forgetting ßÇ₧ßÇÉßÇ¡ßÇòßÇ▒ßÇ╕ßÇüßÇ╗ßÇÇßÇ║ ΓÇö label ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇ₧ßÇ▒ßÇ¼ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» ßÇÇßÇ╝ßÇ¡ßÇ»ßÇòßÇ╝ßÇ▒ßÇ¼ßÇòßÇ╝ßÇÖßÇèßÇ║
    try:
        audit = audit_master_coverage(data_yaml_path)
        if audit.get("ok"):
            empty = audit.get("empty_classes", [])
            if empty:
                print(f"ΓÜá∩╕Å  ßÇ₧ßÇÉßÇ¡ßÇòßÇ╝ßÇ»ßÇ¢ßÇößÇ║ ΓÇö label ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇ¥ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇ₧ßÇ▒ßÇ¼ class {len(empty)} ßÇüßÇ» ßÇ¢ßÇ╛ßÇ¡ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï")
                print(f"   ßÇÑßÇòßÇÖßÇ¼: {', '.join(empty[:12])}{' ...' if len(empty) > 12 else ''}")
                print("   ßÇñ class ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» train ßÇòßÇ╝ßÇ«ßÇ╕ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ model ßÇÇ ßÇÖßÇÖßÇ╛ßÇÉßÇ║ßÇÖßÇ¡ßÇÉßÇ▒ßÇ¼ßÇ╖ßÇòßÇ½ (catastrophic forgetting)ßüï")
                print(f"   ßÇæßÇ¡ßÇ»ßÇ╖ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ╖ßÇ║ freeze={freeze} ßÇûßÇ╝ßÇäßÇ╖ßÇ║ backbone ßÇÇßÇ¡ßÇ» ßÇüßÇ▓ßÇæßÇ¼ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ train ßÇòßÇ½ßÇÖßÇèßÇ║ßüï")
    except Exception:
        pass

    print("=" * 70)
    print("≡ƒÜÇ Continuous Fine-Tuning ßÇàßÇÉßÇäßÇ║ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║...")
    print(f"  Base Model:   {base_model_path}")
    print(f"  Dataset YAML: {data_yaml_path}")
    print(f"  Epochs={epochs}, imgsz={imgsz}, batch={batch}, lr0={lr0}, freeze={freeze}")
    print("=" * 70)

    import time
    from ultralytics import YOLO

    model = YOLO(base_model_path)

    total_epochs = max(1, epochs)

    def _on_epoch_end(trainer: Any) -> None:
        try:
            ep = getattr(trainer, "epoch", 0) + 1
            if on_progress:
                on_progress(ep, total_epochs)
        except Exception:
            pass

    try:
        model.add_callback("on_train_epoch_end", _on_epoch_end)
    except Exception:
        pass

    train_kwargs: Dict[str, Any] = dict(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        name=run_name,
        plots=True,
        project=RUNS_DIR,
    )
    if freeze and int(freeze) > 0:
        train_kwargs["freeze"] = int(freeze)
    results = model.train(**train_kwargs)

    save_dir = getattr(results, "save_dir", None) or os.path.join(RUNS_DIR, run_name)
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    last_pt = os.path.join(save_dir, "weights", "last.pt")

    os.makedirs(MODELS_DIR, exist_ok=True)
    archived: Optional[str] = None
    if os.path.isfile(best_pt):
        archived_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}_best.pt"
        archived = os.path.join(MODELS_DIR, archived_name)
        shutil.copy2(best_pt, archived)

    nc: int = 0
    names_list: List[str] = []
    try:
        m = YOLO(best_pt if os.path.isfile(best_pt) else base_model_path)
        names_attr = getattr(m.model, "names", {})
        if isinstance(names_attr, list):
            names_list = [str(n) for n in names_attr]
        elif isinstance(names_attr, dict):
            names_list = [str(names_attr[i]) for i in sorted(names_attr.keys())]
        nc = len(names_list)
    except Exception:
        pass

    msg = f"Γ£à Training ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï Total {nc} classes ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ best.pt ßÇæßÇ╜ßÇÇßÇ║ßÇ£ßÇ¼ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï"
    print("\n" + "=" * 70)
    print(msg)
    print(f"  Best weights: {best_pt}")
    if archived:
        print(f"  Archived at:  {archived}")
    print("=" * 70)

    return {
        "ok": True,
        "best_pt": best_pt,
        "last_pt": last_pt,
        "archived_pt": archived,
        "total_nc": nc,
        "names": names_list,
        "message": msg,
    }


# ---------------------------------------------------------------------------
# Extra Helper ΓÇö .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇæßÇ▓ßÇ¢ßÇ╛ßÇ¡ class names / nc ßÇÇßÇ¡ßÇ» extract ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ▒ßÇ╕ßÇ¢ßÇößÇ║
# ---------------------------------------------------------------------------
def extract_model_info(pt_path: str) -> Dict[str, Any]:
    """Given a .pt file, return {nc, names, size_kb, path}."""
    if not os.path.isfile(pt_path):
        return {"ok": False, "message": f".pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {pt_path}"}
    try:
        from ultralytics import YOLO
        m = YOLO(pt_path)
        names_attr = getattr(m.model, "names", {})
        if isinstance(names_attr, list):
            names_dict = {i: str(n) for i, n in enumerate(names_attr)}
        elif isinstance(names_attr, dict):
            names_dict = {int(k): str(v) for k, v in names_attr.items()}
        else:
            names_dict = {}
        nc = len(names_dict)
        names = [names_dict[i] for i in sorted(names_dict.keys())]
        size_kb = round(os.path.getsize(pt_path) / 1024.0, 1)
        return {"ok": True, "nc": nc, "names": names, "size_kb": size_kb, "path": pt_path}
    except Exception as e:
        return {"ok": False, "message": f"Model load ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½: {e}"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="VisionSync Master Dataset Builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("add-class", help="data.yaml ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ class ßÇößÇ¼ßÇÖßÇèßÇ║ßÇíßÇ₧ßÇàßÇ║ auto ßÇæßÇèßÇ╖ßÇ║")
    p1.add_argument("--yaml", required=True, help="Master data.yaml path")
    p1.add_argument("--name", required=True, help="Class ßÇößÇ¼ßÇÖßÇèßÇ║ßÇíßÇ₧ßÇàßÇ║")

    p2 = sub.add_parser("merge-dataset", help="Source dataset ßÇÇßÇ¡ßÇ» master ßÇæßÇ▓ merge")
    p2.add_argument("--master", required=True, help="Master dataset root folder")
    p2.add_argument("--source", required=True, help="Source dataset root (data.yaml ßÇòßÇ½ßÇ₧ßÇ▒ßÇ¼ folder)")
    p2.add_argument("--class-name", required=True, help="Master ßÇæßÇ▓ßÇÖßÇ╛ class ßÇößÇ¼ßÇÖßÇèßÇ║")
    p2.add_argument("--source-ids", default=None, help="Source class IDs comma list (optional)")
    p2.add_argument("--mode", default="auto", choices=["auto", "collapse", "per_class"],
                    help="auto=ßÇíßÇ£ßÇ¡ßÇ»ßÇíßÇ£ßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ | collapse=ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÉßÇèßÇ║ßÇ╕ | per_class=ßÇößÇ¼ßÇÖßÇèßÇ║ßÇÉßÇàßÇ║ßÇüßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ╕")

    p3 = sub.add_parser("finetune", help="Base .pt + master dataset ßÇûßÇ╝ßÇäßÇ╖ßÇ║ continuous fine-tune")
    p3.add_argument("--base", required=True, help="Base .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ£ßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕")
    p3.add_argument("--yaml", required=True, help="data.yaml ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ£ßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕")
    p3.add_argument("--epochs", type=int, default=20)
    p3.add_argument("--imgsz", type=int, default=640)
    p3.add_argument("--batch", type=int, default=16)
    p3.add_argument("--lr0", type=float, default=0.001)
    p3.add_argument("--freeze", type=int, default=10,
                    help="Backbone layer ßÇÿßÇÜßÇ║ßÇößÇ╛ßÇàßÇ║ßÇüßÇ» ßÇüßÇ▓ßÇæßÇ¼ßÇ╕ßÇÖßÇ£ßÇ▓ (0 = ßÇÖßÇüßÇ▓)")
    p3.add_argument("--name", default="visionsync_master")

    p4 = sub.add_parser("model-info", help=".pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇæßÇ▓ßÇ¢ßÇ╛ßÇ¡ classes / nc ßÇòßÇ╝ßÇòßÇ▒ßÇ╕ßÇ¢ßÇößÇ║")
    p4.add_argument("--pt", required=True, help=".pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ£ßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕")

    args = parser.parse_args()
    if args.cmd == "add-class":
        res = add_class_to_yaml(args.yaml, args.name)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "merge-dataset":
        ids: Optional[List[int]] = None
        if args.source_ids:
            try:
                ids = [int(x.strip()) for x in args.source_ids.split(",") if x.strip()]
            except Exception:
                print("Γ¥î --source-ids format ßÇÖßÇ╛ßÇ¼ßÇ╕ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï ßÇÑßÇòßÇÖßÇ¼ 0,1,2")
                sys.exit(1)
        res = merge_dataset_into_master(args.master, args.source, args.class_name, ids, args.mode)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    elif args.cmd == "finetune":
        res = continuous_finetune(
            base_model_path=args.base,
            data_yaml_path=args.yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            lr0=args.lr0,
            freeze=args.freeze,
            run_name=args.name,
        )
        print(json.dumps({k: (v if k != "names" else f"[{len(v)} items]")
                          for k, v in res.items()}, ensure_ascii=False, indent=2))
    elif args.cmd == "model-info":
        res = extract_model_info(args.pt)
        if res.get("ok"):
            print(f"nc={res['nc']}   size={res['size_kb']} KB   path={res['path']}")
            print("classes:")
            for i, n in enumerate(res["names"]):  # type: ignore[arg-type]
                print(f"  {i:>3}: {n}")
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
