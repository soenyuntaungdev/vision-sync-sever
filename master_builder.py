"""VisionSync — Master Dataset Builder & Continuous Fine-Tuning Helper

မူလ ၈၀ မျိုး ပါသည့် .pt ဖိုင်နှင့် ပစ္စည်းအသစ်များကို တစ်ခုတည်းသော .pt ဖိုင် အဖြစ်
အလိုအလျောက် ပေါင်းစပ်ပေးသည့် Tools များ

Usage (CLI):
    # ၁။ master dataset ရဲ့ data.yaml ထဲသို့ class အသစ် ထည့်ခြင်း
    python master_builder.py add-class --yaml dataset/master/data.yaml --name my_burmese_coin

    # ၂။ ပစ္စည်းအသစ် (ဥပမာ- Roboflow မှ export လုပ်ထားသော dataset) ကို
    #    master dataset ထဲသို့ label ID auto shift ပြုလုပ်ပြီး merge လုပ်ခြင်း
    python master_builder.py merge-dataset ^
        --master dataset/master ^
        --source dataset/Traffic_Light_v3i_yolov8 ^
        --class-name traffic_light_myanmar

    # ၃။ Base .pt (မူလ ၈၀ မျိုး) + master dataset အသစ် ဖြင့် continuous fine-tune
    python master_builder.py finetune ^
        --base models/my_old_80_classes.pt ^
        --yaml dataset/master/data.yaml ^
        --epochs 20 ^
        --freeze 10 ^
        --name visionsync_master

Usage (Python API):
    from master_builder import add_class_to_yaml, merge_dataset_into_master, continuous_finetune
"""

import argparse
import io
import json
import logging
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
        print("❌ PyYAML မတပ်ဆင်ရသေးပါ။ pip install pyyaml ဖြင့် တပ်ဆင်ပါ။")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _save_yaml(path: str, data: Dict[str, Any]) -> None:
    try:
        import yaml  # type: ignore
    except ImportError:
        print("❌ PyYAML မတပ်ဆင်ရသေးပါ။ pip install pyyaml ဖြင့် တပ်ဆင်ပါ။")
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
        print(f"✅ Master data.yaml အသစ် ဖန်တီးပြီးပါပြီ။ COCO {len(COCO_NAMES)} classes")
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
# Public API 1 — data.yaml ထဲသို့ class အသစ် auto ထည့်ပေးရန်
# ---------------------------------------------------------------------------
def add_class_to_yaml(
    yaml_path: str,
    new_class_name: str,
) -> Dict[str, Any]:
    if not os.path.isfile(yaml_path):
        return {"ok": False, "class_id": -1, "total_nc": -1, "already_exists": False,
                "message": f"data.yaml မတွေ့ရှိပါ: {yaml_path}"}

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
                "message": f"'{name_clean}' သည် ရှိပြီးသား Class ID {cid} ဖြစ်ပါသည်။",
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
        "message": f"အဆင်ပြေပါသည်။ '{name_clean}' ကို Class ID {next_id} အဖြစ် ထည့်သွင်းပြီးပါပြီ။ (Total: {data['nc']})",
    }


# ---------------------------------------------------------------------------
# Public API 2 — Source dataset တစ်ခုကို master dataset ထဲ label shift ပြုလုပ်ပြီး merge
# ---------------------------------------------------------------------------
def _shift_label_file(label_path: str, out_path: str, target_class_id: int,
                      source_nc: int, valid_source_ids: Optional[set]) -> Tuple[int, int]:
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
        if valid_source_ids is not None and cid not in valid_source_ids:
            dropped += 1
            continue
        if cid < 0 or cid >= source_nc:
            dropped += 1
            continue
        
        # FIX: Single class ZIP အများစုအတွက် class 0 ကို target master class id သို့ Direct Mapping ပြုလုပ်ခြင်း
        if source_nc == 1 or cid == 0:
            parts[0] = str(target_class_id)
        else:
            parts[0] = str(target_class_id + cid)

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
) -> Dict[str, Any]:
    _ensure_master_structure(master_root)
    master_yaml = os.path.join(master_root, "data.yaml")
    source_yaml = os.path.join(source_root, "data.yaml")

    try:
        fixup_dataset(source_root, verbose=True)
    except Exception:
        pass
    source_yaml = os.path.join(source_root, "data.yaml")

    if not os.path.isfile(source_yaml):
        return {"ok": False, "message": f"Source data.yaml မတွေ့ရှိပါ: {source_yaml}"}

    src_data = _load_yaml(source_yaml)
    src_names = _normalize_names(src_data.get("names", {}))
    src_nc = int(src_data.get("nc", (max(src_names.keys()) + 1) if src_names else 0))

    add_result = add_class_to_yaml(master_yaml, new_class_name)
    if not add_result["ok"]:
        return {"ok": False, "message": add_result["message"]}
    master_class_id = add_result["class_id"]

    valid_source_ids: Optional[set] = None
    if source_class_ids:
        valid_source_ids = set(source_class_ids)

    splits: List[Tuple[str, str, str, str]] = []
    split_aliases = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    for split, tgt in split_aliases.items():
        a_img = os.path.join(source_root, split, "images")
        a_lbl = os.path.join(source_root, split, "labels")
        if os.path.isdir(a_img) and os.path.isdir(a_lbl):
            splits.append((split, tgt, a_img, a_lbl))
            continue
        b_img = os.path.join(source_root, "images", split)
        b_lbl = os.path.join(source_root, "labels", split)
        if os.path.isdir(b_img) and os.path.isdir(b_lbl):
            splits.append((split, tgt, b_img, b_lbl))
            continue

    if not splits:
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
        return {"ok": False, "message": f"Source ထဲတွင် train/val images/labels folders မတွေ့ရှိပါ။{hint}"}

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
                c, d = _shift_label_file(src_lbl, dst_lbl, master_class_id, src_nc, valid_source_ids)
                total_labels += c
                total_dropped += d
            else:
                open(dst_lbl, "w", encoding="utf-8").close()

    return {
        "ok": True,
        "class_id": master_class_id,
        "class_name": new_class_name,
        "master_nc": add_result["total_nc"],
        "already_exists": add_result["already_exists"],
        "merged_images": total_images,
        "merged_labels": total_labels,
        "dropped_label_rows": total_dropped,
        "message": (
            f"အဆင်ပြေပါသည်။ Class '{new_class_name}' (ID {master_class_id}) အား "
            f"images {total_images} ခု + labels {total_labels} rows ကို master dataset ထဲ merge ပြီးပါပြီ။ "
            f"(total nc={add_result['total_nc']})"
        ),
    }


# ---------------------------------------------------------------------------
# Public API 3 — Continuous fine-tune
# ---------------------------------------------------------------------------
class _LogForwarder(io.StringIO):
    def __init__(self, sink: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._sink = sink
        self._real = sys.stdout

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


class _LoggingToPrint(logging.Handler):
    def __init__(self, sink: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._sink = sink
        self._real_stdout = sys.stdout

    def emit(self, record: Any) -> None:
        try:
            msg = self.format(record)
            if self._sink:
                try:
                    self._sink(msg)
                except Exception:
                    pass
            try:
                self._real_stdout.write(msg + "\n")
                self._real_stdout.flush()
            except Exception:
                pass
        except Exception:
            pass


def continuous_finetune(
    base_model_path: str,
    data_yaml_path: str,
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    lr0: float = 0.001,
    freeze: int = 10,  # FIX: Backbone weights မပျက်စီးအောင် Default freeze=10 ထည့်သွင်းပေးထားပါသည်
    run_name: str = "visionsync_master",
    on_log: Optional[Callable[[str], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError:
        return {"ok": False, "message": "Ultralytics မတပ်ဆင်ရသေးပါ။ pip install ultralytics"}
    import logging as _logging

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_root_handlers: List[Any] = []
    old_root_level: Optional[int] = None
    print_handler: Optional[_LoggingToPrint] = None
    if on_log:
        sys.stdout = _LogForwarder(on_log)
        sys.stderr = _LogForwarder(on_log)
        print_handler = _LoggingToPrint(on_log)
        print_handler.setFormatter(_logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        old_root_level = _logging.root.level
        old_root_handlers = list(_logging.root.handlers)
        for h in old_root_handlers:
            _logging.root.removeHandler(h)
        _logging.root.addHandler(print_handler)
        _logging.root.setLevel(_logging.INFO)
        try:
            import ultralytics  # type: ignore
            for mod_name in ("ultralytics", "yolo"):
                mod_log = _logging.getLogger(mod_name)
                mod_log.addHandler(print_handler)
                mod_log.setLevel(_logging.INFO)
                mod_log.propagate = True
        except Exception:
            pass
    try:
        return _continuous_finetune_inner(
            base_model_path=base_model_path,
            data_yaml_path=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            lr0=lr0,
            freeze=freeze,
            run_name=run_name,
            on_progress=on_progress,
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        if print_handler is not None:
            try:
                _logging.root.removeHandler(print_handler)
                for h in old_root_handlers:
                    _logging.root.addHandler(h)
                if old_root_level is not None:
                    _logging.root.setLevel(old_root_level)
            except Exception:
                pass


def _continuous_finetune_inner(
    base_model_path: str,
    data_yaml_path: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    freeze: int,
    run_name: str,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Any]:
    if not os.path.isfile(base_model_path):
        return {"ok": False, "message": f"Base model မတွေ့ရှိပါ: {base_model_path}"}
    if not os.path.isfile(data_yaml_path):
        return {"ok": False, "message": f"data.yaml မတွေ့ရှိပါ: {data_yaml_path}"}

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
                "message": "Dataset preflight မအောင်မြင်ပါ။ " + " | ".join(pf.get("errors", [])),
                "preflight": pf,
            }
        for w in pf.get("warnings", []) or []:
            print(f"[Warning] {w}")
    except Exception as e:
        return {"ok": False, "message": f"Preflight check error: {e}"}

    print("=" * 70)
    print("🚀 Continuous Fine-Tuning စတင်နေပါသည်...")
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

    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        freeze=freeze,  # FIX: Pretrained feature layers များကို ထိန်းသိမ်းရန် freeze parameter ပို့ပေးခြင်း
        name=run_name,
        plots=True,
        project=RUNS_DIR,
    )

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

    msg = f"✅ Training ပြီးပါပြီ။ Total {nc} classes ပါသည့် best.pt ထွက်လာပါသည်။"
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
# Extra Helper — .pt ဖိုင်ထဲရှိ class names / nc ကို extract လုပ်ပေးရန်
# ---------------------------------------------------------------------------
def extract_model_info(pt_path: str) -> Dict[str, Any]:
    if not os.path.isfile(pt_path):
        return {"ok": False, "message": f".pt ဖိုင်မတွေ့ရှိပါ: {pt_path}"}
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
        return {"ok": False, "message": f"Model load မအောင်မြင်ပါ: {e}"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _cli() -> None:
    parser = argparse.ArgumentParser(description="VisionSync Master Dataset Builder")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("add-class", help="data.yaml ထဲသို့ class နာမည်အသစ် auto ထည့်")
    p1.add_argument("--yaml", required=True, help="Master data.yaml path")
    p1.add_argument("--name", required=True, help="Class နာမည်အသစ်")

    p2 = sub.add_parser("merge-dataset", help="Source dataset ကို master ထဲ merge")
    p2.add_argument("--master", required=True, help="Master dataset root folder")
    p2.add_argument("--source", required=True, help="Source dataset root (data.yaml ပါသော folder)")
    p2.add_argument("--class-name", required=True, help="Master ထဲမှ class နာမည်")
    p2.add_argument("--source-ids", default=None, help="Source class IDs comma list (optional)")

    p3 = sub.add_parser("finetune", help="Base .pt + master dataset ဖြင့် continuous fine-tune")
    p3.add_argument("--base", required=True, help="Base .pt ဖိုင်လမ်းကြောင်း")
    p3.add_argument("--yaml", required=True, help="data.yaml ဖိုင်လမ်းကြောင်း")
    p3.add_argument("--epochs", type=int, default=20)
    p3.add_argument("--imgsz", type=int, default=640)
    p3.add_argument("--batch", type=int, default=16)
    p3.add_argument("--lr0", type=float, default=0.001)
    p3.add_argument("--freeze", type=int, default=10, help="Pre-trained backbone layers freeze ပြုလုပ်မည့် အရေအတွက်")
    p3.add_argument("--name", default="visionsync_master")

    p4 = sub.add_parser("model-info", help=".pt ဖိုင်ထဲရှိ classes / nc ပြပေးရန်")
    p4.add_argument("--pt", required=True, help=".pt ဖိုင်လမ်းကြောင်း")

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
                print("❌ --source-ids format မှားနေပါသည်။ ဥပမာ 0,1,2")
                sys.exit(1)
        res = merge_dataset_into_master(args.master, args.source, args.class_name, ids)
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