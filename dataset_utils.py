"""
VisionSync — Dataset Structure Auto-Fixup Utility
===================================================
Upload / Extract / Training စခင်မှာ ခေါ်သုံးရန် ရည်ရွယ်ချက်။

လုပ်ဆောင်ချက်များ — အလိုအလျောက်:
  1) Flatten nested prefix (ဥပမာ: zip ထဲက dataset/<name>/images → images/)
  2) Roboflow format (train/valid/test) → YOLOv8 standard (images/train, labels/train)
  3) data.yaml ထဲရှိ nc/names/path/train/val/test များ အောင်မြင်စွာ normalize
     • path / train / val / test → absolute paths
     • nc: 0 / names: [] → labels files ထဲမှ ID အများဆုံးတန်ဖိုးဖြင့် rebuild
     • names ရှိပြီး nc မတူ → sync လုပ်ပေးမယ်
  4) Missing folders (images/train, labels/train, images/val, labels/val) များ auto ဖန်တီး

Public API:
  fixup_dataset(root: str, *, verbose=True) -> Dict[str, Any]
"""

import glob
import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================================
# Internal helpers
# =========================================================================

def _read_yaml_simple(path: str) -> Dict[str, Any]:
    """PyYAML မလိုဘဲ ရိုးရှင်းစွာ yaml dict ဖတ်မယ် (roboflow data.yaml အတွက်လုံလောက်တယ်)"""
    data: Dict[str, Any] = {}
    names_list: List[str] = []
    in_names = False
    names_indent = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return {}
    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if in_names:
            if indent <= names_indent and (stripped.startswith("- ") or stripped.startswith("-" * 2) is False and ":" in stripped):
                in_names = False
            elif stripped.startswith("- "):
                val = stripped[2:].strip()
                val = val.strip('"').strip("'")
                names_list.append(val)
                continue
            elif re.match(r"^\s*\d+\s*:\s*", line):
                m = re.match(r"^\s*\d+\s*:\s*(.*)$", line)
                if m:
                    val = m.group(1).strip().strip('"').strip("'")
                    names_list.append(val)
                    continue
                else:
                    in_names = False
            else:
                in_names = False
        if not in_names and ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if val:
                val = val.strip('"').strip("'")
            if key == "names":
                in_names = True
                names_indent = indent
                names_list = []
                if val and not val.startswith("["):
                    pass
                if val.startswith("[") and val.endswith("]"):
                    inner = val[1:-1].strip()
                    if inner:
                        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
                        names_list = [p for p in parts if p]
                continue
            try:
                if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
                    data[key] = int(val)
                elif val.lower() in ("true", "false"):
                    data[key] = val.lower() == "true"
                else:
                    data[key] = val
            except Exception:
                data[key] = val
    if names_list:
        data["names"] = names_list
        if "nc" not in data or not isinstance(data.get("nc"), int) or data["nc"] <= 0:
            data["nc"] = len(names_list)
    return data


def _yaml_safe_str(value: Any) -> str:
    r"""
    String တစ်ခုကို YAML-safe ဖြစ်အောင် normalize လုပ်ပေးမယ်။
    - Windows backslash (\) ကို forward slash (/) နဲ့ replace (path တွေအတွက်၊ Python on Windows က / ကိုလက်ခံတယ်)
    - Double quote (") ကို escape လုပ်ပေးမယ်
    """
    s = str(value)
    s = s.replace("\\", "/")
    s = s.replace('"', '\\"')
    return s


def _yaml_needs_quoting(s: str) -> bool:
    """YAML ရဲ့ value ကို double quote နဲ့ဝိုင်းရမလား ဆုံးဖြတ် (space, :, #, tab, စသဖြင့် ပါရင် quote လိုတယ်)"""
    if not s:
        return True
    specials = [" ", ":", "#", "\t", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`"]
    return any(c in s for c in specials)


def _write_yaml_simple(path: str, data: Dict[str, Any]) -> None:
    """ရိုးရှင်းစွာ yaml dict ရေးမယ်။ (backslash ကို / နဲ့ auto normalize လုပ်ပေးတယ်)"""
    lines: List[str] = []
    order = ["path", "train", "val", "test", "nc", "names"]
    written = set()

    def fmt_scalar(v: Any, prefix: str = "", key: str = "") -> str:
        if isinstance(v, bool):
            return f"{prefix}{key}{'true' if v else 'false'}"
        if isinstance(v, (int, float)):
            return f"{prefix}{key}{v}"
        s = _yaml_safe_str(v)
        line = f"{prefix}{key}"
        if _yaml_needs_quoting(s):
            line += f'"{s}"'
        else:
            line += s
        return line

    for k in order:
        if k in data:
            written.add(k)
            v = data[k]
            if k == "names" and isinstance(v, list):
                lines.append("names:")
                for i, nm in enumerate(v):
                    lines.append(fmt_scalar(nm, prefix="  ", key=f"{i}: "))
            else:
                lines.append(fmt_scalar(v, key=f"{k}: "))

    for k, v in data.items():
        if k in written:
            continue
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for dk, dv in v.items():
                lines.append(fmt_scalar(dv, prefix="  ", key=f"{dk}: "))
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for it in v:
                s = _yaml_safe_str(it)
                if _yaml_needs_quoting(s):
                    lines.append(f'  - "{s}"')
                else:
                    lines.append(f"  - {s}")
        else:
            lines.append(fmt_scalar(v, key=f"{k}: "))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _find_yaml(root: str) -> Optional[str]:
    for nm in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = os.path.join(root, nm)
        if os.path.isfile(p):
            return p
    return None


def _scan_class_ids_in_labels(labels_dirs: List[str]) -> Tuple[int, List[int]]:
    """labels/train, labels/val စတဲ့ folders တွေထဲက class IDs collect လုပ်ပြီး (max_id, all_ids_sorted) ပြန်ပေးမယ်"""
    ids: set = set()
    for d in labels_dirs:
        if not os.path.isdir(d):
            continue
        for txt in glob.glob(os.path.join(d, "*.txt")):
            try:
                with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if not s:
                            continue
                        first = s.split()[0]
                        try:
                            cid = int(float(first))
                            if 0 <= cid < 100000:
                                ids.add(cid)
                        except Exception:
                            pass
            except Exception:
                pass
    if not ids:
        return -1, []
    all_sorted = sorted(ids)
    return all_sorted[-1], all_sorted


def _flatten_nested_dirs(root: str) -> Tuple[bool, str]:
    """
    root အောက်မှာ တစ်ခုပဲရှိတဲ့ subdir ထဲမှာ images/labels/yaml ရှိနေရင် အပြင်ကိုထုတ်ပေးမယ်။
    ပုံစံများ:
      root/
        dataset/
          master/
            images/ ... labels/ ... data.yaml
    => root ထဲမှာ images/labels/data.yaml ကို တိုက်ရိုက်ရွေ့ပြီး dataset/ အားလုံးဖျက်မယ်။
    """
    changed = False
    max_depth = 3
    for _ in range(max_depth):
        items = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]
        yaml_in_root = any(
            nm in files for nm in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
        )
        has_content_in_root = any(nm in items for nm in ("images", "labels", "train", "valid", "val", "test"))
        if yaml_in_root or has_content_in_root:
            break
        if len(items) != 1:
            break
        only = items[0]
        inner = os.path.join(root, only)
        inner_items = os.listdir(inner)
        has_yaml_inner = any(
            nm in inner_items for nm in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
        )
        has_content_inner = any(nm in inner_items for nm in ("images", "labels", "train", "valid", "val", "test"))
        if not (has_yaml_inner or has_content_inner):
            break
        tmp_root = root + "_flatten_tmp"
        if os.path.exists(tmp_root):
            shutil.rmtree(tmp_root, ignore_errors=True)
        shutil.move(inner, tmp_root)
        if os.path.isdir(root):
            for leftover in os.listdir(root):
                lp = os.path.join(root, leftover)
                if os.path.isdir(lp):
                    shutil.rmtree(lp, ignore_errors=True)
                else:
                    try:
                        os.remove(lp)
                    except Exception:
                        pass
        for it in os.listdir(tmp_root):
            shutil.move(os.path.join(tmp_root, it), os.path.join(root, it))
        shutil.rmtree(tmp_root, ignore_errors=True)
        changed = True
    return changed, root


def _convert_roboflow_layout(root: str) -> bool:
    """
    Roboflow export:
      root/train/images, root/train/labels
      root/valid/images, root/valid/labels
      root/test/images,  root/test/labels
    => YOLOv8 standard:
      root/images/train, root/images/val, root/images/test
      root/labels/train, root/labels/val, root/labels/test
    """
    changed = False
    mapping = [
        ("train", "train"),
        ("valid", "val"),
        ("val",   "val"),
        ("test",  "test"),
    ]
    for src_split, dst_split in mapping:
        sp = os.path.join(root, src_split)
        if not os.path.isdir(sp):
            continue
        for sub in ("images", "labels"):
            sdir = os.path.join(sp, sub)
            if not os.path.isdir(sdir):
                continue
            ddir = os.path.join(root, sub, dst_split)
            if os.path.exists(ddir):
                for f in os.listdir(sdir):
                    sfile = os.path.join(sdir, f)
                    dfile = os.path.join(ddir, f)
                    if os.path.isfile(dfile):
                        base, ext = os.path.splitext(f)
                        k = 1
                        while os.path.isfile(os.path.join(ddir, f"{base}_{k}{ext}")):
                            k += 1
                        dfile = os.path.join(ddir, f"{base}_{k}{ext}")
                    shutil.move(sfile, dfile)
                    changed = True
                shutil.rmtree(sdir, ignore_errors=True)
            else:
                os.makedirs(os.path.dirname(ddir), exist_ok=True)
                shutil.move(sdir, ddir)
                changed = True
        left = os.listdir(sp)
        if not left:
            shutil.rmtree(sp, ignore_errors=True)
    return changed


def _ensure_std_dirs(root: str) -> None:
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(root, sub, split), exist_ok=True)


# =========================================================================
# Public API
# =========================================================================

def fixup_dataset(root: str, *, verbose: bool = True) -> Dict[str, Any]:
    """
    Dataset တစ်ခုကို YOLOv8 standard အဖြစ်အောင်မြင်အောင် auto fixup လုပ်ပေးမယ်။

    ပြန်ပေးတဲ့ dict:
      ok: bool
      fixed: list[str] — လုပ်ဆောင်ချက်များစာရင်း
      warnings: list[str]
      yaml_path: str | None
      class_count: int
      layout: str
    """
    result: Dict[str, Any] = {
        "ok": True,
        "fixed": [],
        "warnings": [],
        "yaml_path": None,
        "class_count": 0,
        "layout": "unknown",
    }
    root = os.path.abspath(root)
    if not os.path.isdir(root):
        result["ok"] = False
        result["warnings"].append(f"Root directory မတွေ့ရှိပါ: {root}")
        return result

    # ၁) Nested flatten
    did_flatten, root = _flatten_nested_dirs(root)
    if did_flatten:
        result["fixed"].append("Nested subfolder ထိပ်တန်းကို flatten လုပ်ပြီးပြီ")
    root = os.path.abspath(root)

    # ၂) Roboflow → standard
    did_convert = _convert_roboflow_layout(root)
    if did_convert:
        result["fixed"].append("Roboflow train/valid/test → images/(train|val|test) layout အဖြစ်ပြောင်းပြီးပြီ")

    # ၃) Standard dirs ဖန်တီး
    _ensure_std_dirs(root)

    # ၄) YAML နဲ့ content sync
    yaml_path = _find_yaml(root)
    if yaml_path is None:
        # auto ဖန်တီးပေးမယ် (labels များရှိရင် nc အားလိုက်)
        yaml_path = os.path.join(root, "data.yaml")
        max_cid, _ = _scan_class_ids_in_labels([
            os.path.join(root, "labels", "train"),
            os.path.join(root, "labels", "val"),
            os.path.join(root, "labels", "test"),
        ])
        if max_cid < 0:
            max_cid = 0
        nc = max_cid + 1
        names = [f"class_{i}" for i in range(nc)]
        base = {"path": root,
                "train": os.path.join(root, "images", "train"),
                "val":   os.path.join(root, "images", "val"),
                "test":  os.path.join(root, "images", "test"),
                "nc": nc, "names": names}
        _write_yaml_simple(yaml_path, base)
        result["fixed"].append(f"data.yaml အသစ်ဖန်တီးပြီးပြီ (nc={nc})")
    else:
        cfg = _read_yaml_simple(yaml_path)
        updated = False

        def _pick_split(split: str) -> str:
            """root အောက်မှာရှိတဲ့ standard split folder ကိုရွေးပေးမယ်"""
            alt = os.path.join(abs_root, "images", split)
            if os.path.isdir(alt):
                return alt
            alt2 = os.path.join(abs_root, split)
            return alt2 if os.path.isdir(alt2) else os.path.join(abs_root, "images", split)

        def _is_abs_any_os(p: str) -> bool:
            """Windows drive letter (E:/...) ကိုပါ absolute အဖြစ်သတ်မှတ်ပေးမယ်"""
            if os.path.isabs(p):
                return True
            return re.match(r"^[A-Za-z]:[/\\]", p) is not None

        # path / train / val / test absolute ပြောင်း
        abs_root = root
        for k in ("train", "val", "test", "path"):
            v = cfg.get(k)
            if not isinstance(v, str) or not v:
                continue
            v_norm = v.replace("\\", "/")  # Windows backslashes → /
            if _is_abs_any_os(v_norm):
                # absolute/Windows path — လက်ရှိ machine မှာ တကယ်ရှိမရှိ စစ်ပြီး
                cand = os.path.normpath(v_norm)
                if os.path.isdir(cand):
                    if k == "path" and cfg.get("path") != cand:
                        cfg[k] = cand
                        updated = True
                    continue  # ရှိပြီးသား valid absolute — ထိန်းထား
                # Stale path (ဥပမာ Windows path → Colab) — root layout နဲ့ ပြန်ဆက်
                cfg[k] = abs_root if k == "path" else _pick_split(k)
                result["fixed"].append(f"{k}: မရှိတော့တဲ့ absolute path ({v}) → {cfg[k]}")
                updated = True
                continue
            # relative ဖြစ်ရင် — root နဲ့ စသွားရင် abs လုပ်ပေးမယ်
            cand = os.path.normpath(os.path.join(abs_root, v_norm))
            # သို့သော် Roboflow `../train/images` ကဲ့သို့ ရောက်နိုင်သဖြင့် ရိုးရှင်းစွာ abs လုပ်မယ်
            if k == "path":
                cfg[k] = abs_root
            elif os.path.isdir(cand):
                cfg[k] = cand
            else:
                # standard layout နဲ့ တွဲပေးမယ်
                split_map = {"train": "train", "val": "val", "test": "test"}
                s = split_map.get(k, k)
                cfg[k] = _pick_split(s)
            updated = True

        # nc / names sync
        names = cfg.get("names") if isinstance(cfg.get("names"), list) else []
        nc_cfg = cfg.get("nc") if isinstance(cfg.get("nc"), int) else 0
        if nc_cfg <= 0 and not names:
            max_cid, _ = _scan_class_ids_in_labels([
                os.path.join(root, "labels", "train"),
                os.path.join(root, "labels", "val"),
                os.path.join(root, "labels", "test"),
            ])
            if max_cid < 0:
                max_cid = 0
            nc_new = max_cid + 1
            cfg["nc"] = nc_new
            cfg["names"] = [f"class_{i}" for i in range(nc_new)]
            result["fixed"].append(f"nc:0 / names:[] → labels စစ်ပြီး nc={nc_new} ဖြင့် ပြန်ရေးပြီးပြီ")
            updated = True
        elif nc_cfg <= 0 and names:
            cfg["nc"] = len(names)
            result["fixed"].append(f"nc:0 → names length ({len(names)}) ဖြင့် sync လုပ်ပြီးပြီ")
            updated = True
        elif nc_cfg > 0 and not names:
            cfg["names"] = [f"class_{i}" for i in range(nc_cfg)]
            result["fixed"].append(f"names:[] → nc ({nc_cfg}) အရ auto fill လုပ်ပြီးပြီ")
            updated = True
        elif nc_cfg > 0 and len(names) != nc_cfg:
            if len(names) > nc_cfg:
                names = names[:nc_cfg]
            else:
                names = names + [f"class_{i}" for i in range(len(names), nc_cfg)]
            cfg["names"] = names
            result["fixed"].append(f"names length={len(cfg['names'])} နှင့် nc={nc_cfg} မတူ → sync လုပ်ပြီးပြီ")
            updated = True

        # path ကြားချက် — အမြဲ root absolute နဲ့သာ replace လုပ်ပေးမယ်
        if cfg.get("path") != abs_root:
            cfg["path"] = abs_root
            updated = True

        # test မရှိရင် ဖြည့်ပေးမယ် (မရှိတဲ့ dir ကို မညွှန်ရ — val ကိုသာ fallback)
        if "test" not in cfg or not cfg["test"]:
            test_p = os.path.join(abs_root, "images", "test")
            if os.path.isdir(test_p):
                cfg["test"] = test_p
            elif os.path.isdir(os.path.join(abs_root, "images", "val")):
                cfg["test"] = os.path.join(abs_root, "images", "val")
            else:
                cfg.pop("test", None)
            updated = True

        if updated:
            _write_yaml_simple(yaml_path, cfg)
            result["fixed"].append("data.yaml content ကို normalize လုပ်ပြီးပြီ")

    result["yaml_path"] = yaml_path
    nc_count = 0
    try:
        final_cfg = _read_yaml_simple(yaml_path)
        nc_count = int(final_cfg.get("nc", 0))
    except Exception:
        pass
    result["class_count"] = nc_count

    # layout လက္ခဏာ
    has_im_tr = os.path.isdir(os.path.join(root, "images", "train"))
    has_lb_tr = os.path.isdir(os.path.join(root, "labels", "train"))
    has_rb_tr = os.path.isdir(os.path.join(root, "train", "images"))
    if has_im_tr and has_lb_tr:
        result["layout"] = "yolov8-standard"
    elif has_rb_tr:
        result["layout"] = "roboflow-legacy"
    else:
        result["layout"] = "unknown"

    if verbose and result["fixed"]:
        print(f"[dataset_utils] fixup({os.path.basename(root)}): " + " | ".join(result["fixed"]))
    return result


def preflight_check_training(yaml_path: str) -> Dict[str, Any]:
    """
    Training စခင်မှာ data.yaml path ရှိမရှိ, train/val images/labels တွေရှိမရှိ၊
    nc / names တွေကိုက်ညီမညီ စစ်ပေးမယ်။
    """
    out: Dict[str, Any] = {"ok": True, "errors": [], "warnings": [], "info": {}}
    if not os.path.isabs(yaml_path):
        yaml_path = os.path.join(BACKEND_DIR, yaml_path)
    if not os.path.isfile(yaml_path):
        out["ok"] = False
        out["errors"].append(f"data.yaml မတွေ့ရှိပါ: {yaml_path}")
        return out
    # စစ်ဆေးခြင်းမစမီ yaml paths များကို ဤ machine အတွက် auto ပြန်ပြင်ပေးမယ်
    # (Windows absolute path ပါဝင်သော data.yaml ကို Colab/Linux မှာ run နိုင်ရန်)
    try:
        fixup_dataset(os.path.dirname(os.path.abspath(yaml_path)), verbose=False)
    except Exception:
        pass
    cfg = _read_yaml_simple(yaml_path)
    root = cfg.get("path") or os.path.dirname(os.path.abspath(yaml_path))
    out["info"]["yaml"] = yaml_path
    out["info"]["root"] = root
    # train/val စစ်
    for k in ("train", "val"):
        p = cfg.get(k)
        if isinstance(p, str) and p:
            if not os.path.isabs(p):
                p = os.path.normpath(os.path.join(root, p))
        else:
            out["errors"].append(f"data.yaml ထဲမှာ {k}: path မရှိပါ")
            continue
        out["info"][f"{k}_images"] = p
        if not os.path.isdir(p):
            out["ok"] = False
            out["errors"].append(f"{k} images directory မတွေ့ရှိပါ: {p}")
            continue
        n_imgs = len([x for x in os.listdir(p) if not x.lower().startswith(".")])
        out["info"][f"{k}_count"] = n_imgs
        if n_imgs == 0:
            # ပုံမရှိဘဲ train လုပ်ရင် Ultralytics က ဘာမှမသင်ရဘဲ ဖြတ်သွားပြီး
            # ဘာမှ detect မလုပ်တဲ့ .pt တစ်ခု ထွက်လာသည်။ ဒါကို ကြိုတားမည်။
            out["errors"].append(f"{k} images folder ({p}) မှာ ပုံ တစ်ပုံမှမရှိပါ")
        # label folder ရှာမယ်
        p_parts = os.path.normpath(p).split(os.sep)
        if "images" in p_parts:
            label_p = os.sep.join(["labels" if x == "images" else x for x in p_parts])
            if not os.path.isdir(label_p):
                out["errors"].append(f"Corresponding labels directory မတွေ့ရှိပါ: {label_p}")
            else:
                n_lbl = len([x for x in os.listdir(label_p) if not x.lower().startswith(".")])
                out["info"][f"{k}_labels"] = label_p
                out["info"][f"{k}_labels_count"] = n_lbl
                if n_lbl == 0:
                    out["warnings"].append(f"{k} labels folder မှာ .txt မရှိသေးပါ")
    nc = cfg.get("nc") if isinstance(cfg.get("nc"), int) else 0
    names = cfg.get("names") if isinstance(cfg.get("names"), list) else []
    out["info"]["nc"] = nc
    out["info"]["names_count"] = len(names)
    if nc <= 0:
        out["errors"].append(f"nc={nc} ကို မှန်ကန်အောင် ပြင်ပေးရပါမည်")
    elif names and len(names) != nc:
        out["warnings"].append(f"nc={nc} နှင့် names count={len(names)} မတူပါ")

    # Label ဖိုင်တွေ လုံးဝမရှိရင် train လုပ်လို့ရပေမယ့် ဘာမှ မသင်ရသဖြင့်
    # ရလာသော .pt က ဘာမှမမြင်တော့ပါ။ ဒါကို error အဖြစ် သတ်မှတ်သည်။
    for k in ("train", "val"):
        if out["info"].get(f"{k}_labels_count") == 0 and out["info"].get(f"{k}_count", 0) > 0:
            out["errors"].append(
                f"{k} labels folder ထဲမှာ .txt ဖိုင် တစ်ခုမှမရှိပါ — "
                "ဒီအတိုင်း train ရင် ဘာမှ detect မလုပ်တဲ့ model ထွက်လာပါမည်"
            )

    # errors တစ်ခုခုရှိရင် ok ကို အမြဲ False ဖြစ်စေရမည်။
    # (ယခင်က အချို့ error branch တွေမှာ ok=False မလုပ်ခဲ့လို့ ပျက်နေသော dataset နဲ့
    #  training စထွက်သွားပြီး၊ ရလာတဲ့ .pt က အလုပ်မလုပ်တာ ဖြစ်ခဲ့သည်။)
    out["ok"] = len(out["errors"]) == 0
    if out["ok"]:
        out["info"]["ready"] = True
    return out
