import io
import os
import shutil
import sys
import threading
import time
import zipfile
from typing import Any, Dict, List, Optional

from dataset_utils import fixup_dataset, preflight_check_training


BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
RUNS_DIR = os.path.join(BACKEND_DIR, "runs", "detect")
MODELS_DIR = os.path.join(BACKEND_DIR, "models")
UPLOADS_DIR = os.path.join(BACKEND_DIR, "uploads")

# Active model ßÇÇßÇ¡ßÇ» source code (main.py) ßÇæßÇ▓ ßÇòßÇ╝ßÇößÇ║ßÇ¢ßÇ▒ßÇ╕ßÇÖßÇèßÇ╖ßÇ║ßÇíßÇàßÇ¼ßÇ╕ ßÇñ JSON ßÇæßÇ▓ ßÇÖßÇ╛ßÇÉßÇ║ßÇ₧ßÇèßÇ║ßüï
# Server restart / Colab cell ßÇòßÇ╝ßÇößÇ║ run ßÇ£ßÇ»ßÇòßÇ║ßÇ£ßÇèßÇ║ßÇ╕ ßÇíßÇ₧ßÇÇßÇ║ßÇ¥ßÇäßÇ║ßÇößÇ▒ßÇàßÇ▒ßÇ¢ßÇößÇ║ßüï
ACTIVE_MODEL_FILE = os.path.join(BACKEND_DIR, "active_model.json")
DEFAULT_MODEL = "yolov8m-oiv7.pt"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


def _main_globals() -> Dict[str, Any]:
    """
    ßÇÉßÇÇßÇÜßÇ║ run ßÇößÇ▒ßÇ₧ßÇ▒ßÇ¼ main module ßüÅ global namespace ßÇÇßÇ¡ßÇ» ßÇ¢ßÇÜßÇ░ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    `uvicorn main:app` ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║ sys.modules["main"]ßüè `python main.py` ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║
    sys.modules["__main__"] ßÇûßÇ╝ßÇàßÇ║ßÇ₧ßÇèßÇ║ßüï ßÇÆßÇ«ßÇößÇ╛ßÇàßÇ║ßÇüßÇ»ßÇàßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÖßÇ╛ßÇ¼ `app` + `detector` ßÇ¢ßÇ╛ßÇ¡ßÇÉßÇ▓ßÇ╖ßÇƒßÇ¼ßÇÇßÇ¡ßÇ»
    ßÇ¢ßÇ╜ßÇ▒ßÇ╕ßÇ¢ßÇÖßÇèßÇ║ßüï ßÇÖßÇƒßÇ»ßÇÉßÇ║ßÇ¢ßÇäßÇ║ main.py ßÇÇßÇ¡ßÇ» ßÇÆßÇ»ßÇÉßÇ¡ßÇÜßÇíßÇÇßÇ╝ßÇ¡ßÇÖßÇ║ import ßÇÖßÇ¡ßÇòßÇ╝ßÇ«ßÇ╕ßüè ßÇÉßÇÇßÇÜßÇ║ serve ßÇößÇ▒ßÇÉßÇ▓ßÇ╖
    module ßÇÖßÇƒßÇ»ßÇÉßÇ║ßÇÉßÇ▓ßÇ╖ copy ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼ detector ßÇÇßÇ¡ßÇ» ßÇ£ßÇ▓ßÇÖßÇ¡ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇ£ßÇ¡ßÇ»ßÇ╖ live reload ßÇÇ
    "ßÇûßÇ╝ßÇàßÇ║ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇòßÇ╝ßÇ«" ßÇ£ßÇ¡ßÇ»ßÇ╖ßÇòßÇ╝ßÇòßÇ▒ßÇÖßÇÜßÇ╖ßÇ║ ßÇÿßÇ¼ßÇÖßÇ╛ßÇÖßÇòßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ßÇÿßÇ░ßÇ╕ ßÇûßÇ╝ßÇàßÇ║ßÇÉßÇÉßÇ║ßÇ₧ßÇèßÇ║ßüï
    """
    for key in ("main", "__main__"):
        mod = sys.modules.get(key)
        if mod is not None and hasattr(mod, "detector") and hasattr(mod, "app"):
            return mod.__dict__
    for key in ("main", "__main__"):
        mod = sys.modules.get(key)
        if mod is not None and hasattr(mod, "detector"):
            return mod.__dict__
    raise RuntimeError(
        "Serve ßÇößÇ▒ßÇ₧ßÇ▒ßÇ¼ main module ßÇÇßÇ¡ßÇ» ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½ (detector global ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½)ßüï "
        "Server ßÇÇßÇ¡ßÇ» `uvicorn main:app` ßÇûßÇ╝ßÇäßÇ╖ßÇ║ run ßÇæßÇ¼ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ ßÇàßÇàßÇ║ßÇòßÇ½ßüï"
    )


def read_active_model() -> str:
    """active_model.json ßÇæßÇ▓ßÇÇ ßÇÖßÇ╛ßÇÉßÇ║ßÇæßÇ¼ßÇ╕ßÇ₧ßÇ▒ßÇ¼ model path (ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇ¢ßÇäßÇ║ default) ßÇÇßÇ¡ßÇ» ßÇòßÇ╝ßÇößÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï"""
    try:
        import json
        with open(ACTIVE_MODEL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        path = str(data.get("model_path") or "").strip()
        if not path:
            return DEFAULT_MODEL
        full = path if os.path.isabs(path) else os.path.join(BACKEND_DIR, path)
        if os.path.isfile(full):
            return full
        # ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇößÇ▒ßÇ¢ßÇäßÇ║ (ßÇÑßÇòßÇÖßÇ¼ Colab session ßÇíßÇ₧ßÇàßÇ║) default ßÇÇßÇ¡ßÇ» ßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇÖßÇèßÇ║
        return DEFAULT_MODEL
    except Exception:
        return DEFAULT_MODEL


def write_active_model(model_path: str, extra: Optional[Dict[str, Any]] = None) -> None:
    import json
    payload: Dict[str, Any] = {"model_path": model_path, "updated_at": time.time()}
    if extra:
        payload.update(extra)
    with open(ACTIVE_MODEL_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


class TrainingLogCapture(io.StringIO):
    """
    Training log ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» buffer ßÇæßÇ▓ ßÇàßÇ»ßÇ₧ßÇ¡ßÇÖßÇ║ßÇ╕ßÇòßÇ╝ßÇ«ßÇ╕ ßÇÉßÇÇßÇÜßÇ╖ßÇ║ console ßÇÇßÇ¡ßÇ»ßÇòßÇ½ ßÇåßÇÇßÇ║ßÇòßÇ¡ßÇ»ßÇ╖ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    (ßÇÜßÇüßÇäßÇ║ßÇÇ console ßÇÇßÇ¡ßÇ» ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇ¥ßÇÖßÇòßÇ¡ßÇ»ßÇ╖ßÇüßÇ▓ßÇ╖ßÇ£ßÇ¡ßÇ»ßÇ╖ Colab ßÇÖßÇ╛ßÇ¼ training ßÇàßÇÉßÇ¼ßÇößÇ▓ßÇ╖ uvicorn ßÇ¢ßÇ▓ßÇ╖
     output ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ ßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇüßÇ▓ßÇ╖ßÇ₧ßÇèßÇ║ßüï)
    """

    def __init__(self, log_buffer: List[str]):
        super().__init__()
        self.log_buffer = log_buffer
        self._stdout = sys.__stdout__

    def write(self, s: str) -> int:
        stripped = s.rstrip("\n")
        if stripped:
            self.log_buffer.append(stripped)
        try:
            if self._stdout:
                self._stdout.write(s)
                self._stdout.flush()
        except Exception:
            pass
        return len(s)

    def flush(self) -> None:
        try:
            if self._stdout:
                self._stdout.flush()
        except Exception:
            pass


class TrainingManager:
    _instance: Optional["TrainingManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "TrainingManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.status: str = "idle"
        self.current_config: Dict[str, Any] = {}
        self.logs: List[str] = []
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.progress: int = 0
        self.error: Optional[str] = None
        self.result_weights: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._result_artifact: Optional[Dict[str, Any]] = None

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets: List[Dict[str, Any]] = []
        if not os.path.isdir(DATASET_DIR):
            return datasets
        for name in sorted(os.listdir(DATASET_DIR)):
            full = os.path.join(DATASET_DIR, name)
            if not os.path.isdir(full):
                continue
            # Auto fixup ßÇ¢ßÇ╛ßÇ▒ßÇ╕ßÇƒßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ dataset ßÇÉßÇ╜ßÇ▒ßÇÇßÇ¡ßÇ»ßÇ£ßÇèßÇ║ßÇ╕ ßÇòßÇ╝ßÇößÇ║ßÇòßÇ╝ßÇäßÇ║ßÇòßÇ▒ßÇ╕ßÇÖßÇÜßÇ║
            try:
                fixup_dataset(full, verbose=False)
            except Exception:
                pass
            yaml_candidates = [
                os.path.join(full, "data.yaml"),
                os.path.join(full, "data.yml"),
                os.path.join(full, "dataset.yaml"),
                os.path.join(full, "dataset.yml"),
            ]
            yaml_path = None
            for c in yaml_candidates:
                if os.path.isfile(c):
                    yaml_path = c
                    break
            if yaml_path:
                datasets.append({
                    "name": name,
                    "yaml_path": os.path.relpath(yaml_path, BACKEND_DIR),
                    "class_count": self._count_classes(yaml_path),
                })
        return datasets

    def _count_classes(self, yaml_path: str) -> int:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            m = re.search(r"nc:\s*(\d+)", content)
            if m:
                return int(m.group(1))
        except Exception:
            pass
        return -1

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        search_dirs = [RUNS_DIR, MODELS_DIR, BACKEND_DIR]
        seen = set()
        for base in search_dirs:
            if not os.path.isdir(base):
                continue
            for root, _, files in os.walk(base):
                for f in sorted(files):
                    if f.endswith(".pt"):
                        full = os.path.join(root, f)
                        rel = os.path.relpath(full, BACKEND_DIR)
                        if rel in seen:
                            continue
                        seen.add(rel)
                        try:
                            mtime = os.path.getmtime(full)
                            size_kb = round(os.path.getsize(full) / 1024, 1)
                        except Exception:
                            mtime = 0
                            size_kb = 0
                        models.append({
                            "name": f,
                            "path": rel.replace("\\", "/"),
                            "modified": mtime,
                            "size_kb": size_kb,
                        })
        return sorted(models, key=lambda m: m["modified"], reverse=True)

    def current_model(self) -> Optional[str]:
        """ßÇ£ßÇÇßÇ║ßÇ¢ßÇ╛ßÇ¡ serve ßÇößÇ▒ßÇ₧ßÇ▒ßÇ¼ detector ßÇ¢ßÇ▓ßÇ╖ model pathßüï ßÇÖßÇ¢ßÇ¢ßÇäßÇ║ ßÇÖßÇ╛ßÇÉßÇ║ßÇæßÇ¼ßÇ╕ßÇÉßÇ▓ßÇ╖ JSON ßÇÇßÇ¡ßÇ» ßÇÇßÇ╝ßÇèßÇ╖ßÇ║ßÇÖßÇèßÇ║ßüï"""
        try:
            det = _main_globals().get("detector")
            name = getattr(det, "model_name", None)
            if name:
                return self._display_path(str(name))
        except Exception:
            pass
        return self._display_path(read_active_model())

    @staticmethod
    def _display_path(path: str) -> str:
        """BACKEND_DIR ßÇíßÇ▒ßÇ¼ßÇÇßÇ║ßÇÇ path ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║ relative ßÇíßÇûßÇ╝ßÇàßÇ║ ßÇòßÇ╝ßÇòßÇ▒ßÇ╕ßÇÖßÇèßÇ║ (UI ßÇößÇ╛ßÇäßÇ╖ßÇ║ ßÇÇßÇ¡ßÇ»ßÇÇßÇ║ßÇ¢ßÇößÇ║)ßüï"""
        try:
            if os.path.isabs(path):
                rel = os.path.relpath(path, BACKEND_DIR)
                if not rel.startswith(".."):
                    return rel.replace("\\", "/")
        except Exception:
            pass
        return path.replace("\\", "/")

    def activate_model(self, model_path: str) -> Dict[str, Any]:
        """
        .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÇßÇ¡ßÇ» ßÇ£ßÇÇßÇ║ßÇ¢ßÇ╛ßÇ¡ detection model ßÇíßÇûßÇ╝ßÇàßÇ║ ßÇÉßÇÇßÇÜßÇ║ load ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ßÇÖßÇ╛ swap ßÇ£ßÇ»ßÇòßÇ║ßÇ₧ßÇèßÇ║ßüï

        ßÇíßÇ¢ßÇ▒ßÇ╕ßÇÇßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇèßÇ║ßÇÖßÇ╛ßÇ¼ ΓÇö load ßÇÖßÇ¢ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ "ok" ßÇƒßÇ» ßÇÿßÇÜßÇ║ßÇÉßÇ▒ßÇ¼ßÇ╖ßÇÖßÇ╛ ßÇÖßÇòßÇ╝ßÇößÇ║ßÇòßÇ½ßüï (ßÇÜßÇüßÇäßÇ║ code ßÇÇ
        ObjectDetector ßÇ¢ßÇ▓ßÇ╖ silent fallback ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ╖ßÇ║ load ßÇÖßÇ¢ßÇ£ßÇèßÇ║ßÇ╕ ßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇƒßÇ»ßÇòßÇ╝ßÇòßÇ╝ßÇ«ßÇ╕
        random fake detection ßÇÉßÇ╜ßÇ▒ ßÇæßÇ»ßÇÉßÇ║ßÇößÇ▒ßÇüßÇ▓ßÇ╖ßÇ₧ßÇèßÇ║ßüï)
        """
        full = model_path if os.path.isabs(model_path) else os.path.join(BACKEND_DIR, model_path)
        full = os.path.normpath(full)
        if not os.path.isfile(full):
            return {"ok": False, "error": f"Model file not found: {model_path}"}
        if not full.lower().endswith(".pt"):
            return {"ok": False, "error": f".pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ₧ßÇ¼ activate ßÇ£ßÇ»ßÇòßÇ║ßÇ£ßÇ¡ßÇ»ßÇ╖ßÇ¢ßÇòßÇ½ßÇ₧ßÇèßÇ║: {model_path}"}

        # ßüüßüï ßÇíßÇ₧ßÇàßÇ║ßÇÇßÇ¡ßÇ» strict mode ßÇößÇ▓ßÇ╖ load ΓÇö ßÇÖßÇ¢ßÇ¢ßÇäßÇ║ ßÇÆßÇ«ßÇößÇ▒ßÇ¢ßÇ¼ßÇÖßÇ╛ßÇ¼ßÇòßÇ▓ ßÇ¢ßÇòßÇ║ßÇÖßÇÜßÇ║ßüè ßÇíßÇƒßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ßÇÇ ßÇåßÇÇßÇ║ßÇíßÇ£ßÇ»ßÇòßÇ║ßÇ£ßÇ»ßÇòßÇ║ßÇößÇ▒ßÇÖßÇèßÇ║
        try:
            from detector import ObjectDetector
            new_detector = ObjectDetector(model_name=full, allow_fallback=False)
        except Exception as e:
            return {
                "ok": False,
                "error": (
                    f"Model load ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½ ({type(e).__name__}): {e}. "
                    "Ultralytics/torch version ßÇÖßÇÇßÇ¡ßÇ»ßÇÇßÇ║ßÇÉßÇ¼ (ßÇ₧ßÇ¡ßÇ»ßÇ╖) .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ ßÇòßÇ╗ßÇÇßÇ║ßÇößÇ▒ßÇÉßÇ¼ ßÇûßÇ╝ßÇàßÇ║ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï"
                ),
            }
        if new_detector.use_fallback or new_detector.model is None:
            return {"ok": False, "error": f"Model load ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½: {new_detector.load_error}"}

        # ßüéßüï Serve ßÇößÇ▒ßÇ₧ßÇ▒ßÇ¼ main module ßÇæßÇ▓ßÇÇ detector ßÇÇßÇ¡ßÇ» ßÇ£ßÇ▓ßÇÖßÇèßÇ║
        try:
            g = _main_globals()
        except Exception as e:
            return {"ok": False, "error": f"Live reload ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½: {e}"}
        prev = g.get("detector")
        prev_name = getattr(prev, "model_name", "?")
        g["detector"] = new_detector

        # ßüâßüï Restart ßÇ£ßÇèßÇ║ßÇ╕ ßÇíßÇ₧ßÇÇßÇ║ßÇ¥ßÇäßÇ║ßÇößÇ▒ßÇíßÇ▒ßÇ¼ßÇäßÇ║ ßÇÖßÇ╛ßÇÉßÇ║ßÇæßÇ¼ßÇ╕ßÇÖßÇèßÇ║
        persisted = True
        persist_err = ""
        try:
            write_active_model(
                self._display_path(full),
                {"nc": len(new_detector.class_names), "names": new_detector.class_names},
            )
        except Exception as e:
            persisted = False
            persist_err = str(e)

        return {
            "ok": True,
            "model_path": self._display_path(full),
            "nc": len(new_detector.class_names),
            "names": new_detector.class_names,
            "persisted": persisted,
            "message": (
                f"Γ£à Activated: {self._display_path(full)} "
                f"(nc={len(new_detector.class_names)}, ßÇíßÇ¢ßÇäßÇ║: {prev_name}) ΓÇö "
                "restart ßÇÖßÇ£ßÇ¡ßÇ»ßÇÿßÇ▓ ßÇüßÇ╗ßÇÇßÇ║ßÇüßÇ╗ßÇäßÇ║ßÇ╕ ßÇíßÇ₧ßÇÇßÇ║ßÇ¥ßÇäßÇ║ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï"
                + ("" if persisted else f" ΓÜá∩╕Å active_model.json ßÇ₧ßÇ¡ßÇÖßÇ║ßÇ╕ßÇÖßÇ¢ßÇòßÇ½: {persist_err}")
            ),
        }

    def extract_uploaded_zip(self, zip_filepath: str, dataset_name: str) -> Dict[str, Any]:
        target_dir = os.path.join(DATASET_DIR, dataset_name)
        os.makedirs(target_dir, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_filepath, "r") as zf:
                members = zf.namelist()
                common_prefix = os.path.commonpath(members) if members else ""
                extract_root = target_dir
                if common_prefix and all(m.startswith(common_prefix) for m in members):
                    extract_root = os.path.dirname(target_dir)
                    target_dir_final = target_dir
                    extracted = []
                    for m in members:
                        if m.endswith("/"):
                            continue
                        rel_to_prefix = os.path.relpath(m, common_prefix)
                        out_path = os.path.join(target_dir_final, rel_to_prefix)
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        with zf.open(m) as src, open(out_path, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted.append(rel_to_prefix)
                    res = {"ok": True, "files": len(extracted), "target_dir": os.path.relpath(target_dir, BACKEND_DIR)}
                else:
                    zf.extractall(extract_root)
                    res = {"ok": True, "target_dir": os.path.relpath(target_dir, BACKEND_DIR)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        # Auto fixup dataset structure / yaml
        try:
            fix_info = fixup_dataset(target_dir, verbose=True)
            res["fixup"] = {
                "fixed": fix_info.get("fixed", []),
                "warnings": fix_info.get("warnings", []),
                "layout": fix_info.get("layout", "unknown"),
                "class_count": fix_info.get("class_count", 0),
            }
        except Exception as e:
            res["fixup_error"] = str(e)
        return res

    def get_state(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "config": self.current_config,
            "logs": self.logs,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "error": self.error,
            "result_weights": self.result_weights,
            "duration_sec": None if self.started_at is None else (time.time() - self.started_at) if self.finished_at is None else (self.finished_at - self.started_at),
        }

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.status == "running":
            return {"ok": False, "error": "Training already in progress"}

        if self.status == "stopping":
            return {"ok": False, "error": "ßÇ¢ßÇ╛ßÇ▒ßÇ╖ßÇÇ training ßÇ¢ßÇòßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇòßÇ½ßüï ßÇüßÇÅßÇàßÇ▒ßÇ¼ßÇäßÇ╖ßÇ║ßÇòßÇ½ßüï"}

        data_yaml_rel = config.get("data_yaml_path", "dataset/data.yaml")
        data_yaml_path = os.path.join(BACKEND_DIR, data_yaml_rel) if not os.path.isabs(data_yaml_rel) else data_yaml_rel
        if not os.path.isfile(data_yaml_path):
            return {"ok": False, "error": f"data.yaml not found: {data_yaml_rel}"}

        # Preflight validation: structure, paths, nc/names ßÇÉßÇ╜ßÇ▒ ßÇàßÇàßÇ║ßÇòßÇ▒ßÇ╕ßÇÖßÇÜßÇ║
        try:
            check = preflight_check_training(data_yaml_path)
            if not check.get("ok", False):
                errs = check.get("errors", [])
                return {
                    "ok": False,
                    "error": "Dataset preflight ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½ßüï " + " | ".join(errs),
                    "preflight": check,
                }
            # Warnings ßÇÉßÇ╜ßÇ▒ßÇ¢ßÇ╛ßÇ¡ßÇ¢ßÇäßÇ║ logs ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼ßÇÖßÇ╛ßÇÉßÇ║ßÇæßÇ¼ßÇ╕ßÇÖßÇÜßÇ║
            self._pending_preflight = check
        except Exception as e:
            return {"ok": False, "error": f"Preflight check error: {e}"}

        self.status = "running"
        self.current_config = config
        self.logs = []
        self.started_at = time.time()
        self.finished_at = None
        self.progress = 0
        self.error = None
        self.result_weights = None
        self._result_artifact = None
        self._stop_event.clear()

        def runner(cfg: Dict[str, Any], yaml_path: str) -> None:
            import traceback
            epochs = int(cfg.get("epochs", 50))
            img_size = int(cfg.get("img_size", 640))
            batch_size = int(cfg.get("batch_size", 16))
            base_model = cfg.get("base_model", "yolov8n.pt")
            run_name = cfg.get("run_name", "visionsync_custom")
            freeze = cfg.get("freeze", None)

            log_capture = TrainingLogCapture(self.logs)
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = log_capture
            sys.stderr = log_capture
            try:
                self.logs.append(f"[VisionSync] Starting training: {run_name}")
                self.logs.append(f"  data_yaml = {yaml_path}")
                self.logs.append(f"  base_model = {base_model}")
                self.logs.append(f"  epochs = {epochs}, imgsz = {img_size}, batch = {batch_size}")
                pf = getattr(self, "_pending_preflight", None)
                if isinstance(pf, dict):
                    info = pf.get("info", {})
                    if isinstance(info, dict) and info:
                        self.logs.append(f"[VisionSync] Preflight info: train_imgs={info.get('train_count','?')}, val_imgs={info.get('val_count','?')}, nc={info.get('nc','?')}")
                    for w in pf.get("warnings", []) or []:
                        self.logs.append(f"[VisionSync] ΓÜá∩╕Å  Preflight warning: {w}")
                from ultralytics import YOLO

                base_path = base_model
                if not os.path.isabs(base_path):
                    candidate = os.path.join(BACKEND_DIR, base_path)
                    if os.path.isfile(candidate):
                        base_path = candidate
                model = YOLO(base_path)

                class CallbackHook:
                    pass

                total_epochs = max(1, epochs)
                def on_train_epoch_end(trainer: Any) -> None:
                    try:
                        epoch = getattr(trainer, "epoch", 0) + 1
                        self.progress = int(min(100, (epoch / total_epochs) * 100))
                    except Exception:
                        pass

                try:
                    model.add_callback("on_train_epoch_end", on_train_epoch_end)
                except Exception:
                    pass

                train_kwargs: Dict[str, Any] = dict(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    name=run_name,
                    plots=True,
                    project=os.path.join(BACKEND_DIR, "runs", "detect"),
                )
                if freeze is not None:
                    train_kwargs["freeze"] = int(freeze)
                    self.logs.append(f"  freeze = {int(freeze)} layers (backbone ßÇÇßÇ¡ßÇ» ßÇæßÇ¡ßÇößÇ║ßÇ╕ßÇæßÇ¼ßÇ╕ßÇÖßÇèßÇ║)")
                results = model.train(**train_kwargs)

                best_path = None
                try:
                    save_dir = getattr(results, "save_dir", None) or os.path.join(BACKEND_DIR, "runs", "detect", run_name)
                    candidate = os.path.join(save_dir, "weights", "best.pt")
                    if os.path.isfile(candidate):
                        best_path = candidate
                except Exception:
                    pass

                if best_path:
                    try:
                        archive_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}_best.pt"
                        archive_path = os.path.join(MODELS_DIR, archive_name)
                        shutil.copy2(best_path, archive_path)
                        self.result_weights = os.path.relpath(archive_path, BACKEND_DIR).replace("\\", "/")
                    except Exception:
                        self.result_weights = os.path.relpath(best_path, BACKEND_DIR).replace("\\", "/")

                self.logs.append("[VisionSync] Training finished successfully.")
                self.logs.append(f"[VisionSync] Best weights: {self.result_weights}")
                self.progress = 100
                self.status = "completed"
            except Exception as e:
                tb = traceback.format_exc()
                self.logs.append(f"[VisionSync] TRAINING ERROR: {e}")
                self.logs.extend(tb.splitlines())
                self.error = str(e)
                self.status = "failed"
            finally:
                self.finished_at = time.time()
                # ßÇÉßÇüßÇ╝ßÇ¼ßÇ╕ job ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÇ stdout ßÇÇßÇ¡ßÇ» ßÇÇßÇ╝ßÇ¼ßÇ╕ßÇûßÇ╝ßÇÉßÇ║ßÇÜßÇ░ßÇæßÇ¼ßÇ╕ßÇ¢ßÇäßÇ║ ßÇíßÇ▓ßÇÆßÇ½ßÇÇßÇ¡ßÇ» ßÇÖßÇûßÇ╗ßÇÇßÇ║ßÇÖßÇ¡ßÇàßÇ▒ßÇ¢ßÇößÇ║
                # ßÇÇßÇ¡ßÇ»ßÇÜßÇ╖ßÇ║ capture object ßÇíßÇÉßÇ¡ßÇ»ßÇäßÇ║ßÇ╕ ßÇÇßÇ╗ßÇößÇ║ßÇößÇ▒ßÇÖßÇ╛ßÇ₧ßÇ¼ ßÇòßÇ╝ßÇößÇ║ßÇæßÇ¼ßÇ╕ßÇÖßÇèßÇ║ßüï
                if sys.stdout is log_capture:
                    sys.stdout = old_stdout
                if sys.stderr is log_capture:
                    sys.stderr = old_stderr

        self._thread = threading.Thread(target=runner, args=(config, data_yaml_path), daemon=True)
        self._thread.start()
        return {"ok": True, "message": "Training started"}

    def stop_training(self) -> Dict[str, Any]:
        if self.status != "running":
            return {"ok": False, "error": "No training running"}
        self._stop_event.set()
        self.status = "stopping"
        self.logs.append("[VisionSync] Stop requested (terminating training process on next update)...")
        try:
            import signal
            os.kill(os.getpid(), signal.SIGINT)
        except Exception:
            self.status = "stopped"
            self.finished_at = time.time()
        return {"ok": True, "message": "Stop signal sent"}
