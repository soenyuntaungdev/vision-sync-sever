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

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)


class TrainingLogCapture(io.StringIO):
    def __init__(self, log_buffer: List[str]):
        super().__init__()
        self.log_buffer = log_buffer
        self._stdout = sys.stdout
        self._stderr = sys.stderr

    def write(self, s: str) -> int:
        stripped = s.rstrip("\n")
        if stripped:
            self.log_buffer.append(stripped)
        return len(s)

    def flush(self) -> None:
        if self._stdout:
            self._stdout.flush()


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
            # Auto fixup ရှေးဟောင်း dataset တွေကိုလည်း ပြန်ပြင်ပေးမယ်
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
        detector_path = os.path.join(BACKEND_DIR, "main.py")
        try:
            with open(detector_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            m = re.search(r"ObjectDetector\(model_name=[\"']([^\"']+)[\"']\)", content)
            if m:
                return m.group(1)
        except Exception:
            pass
        return None

    def activate_model(self, model_path: str) -> Dict[str, Any]:
        full = os.path.join(BACKEND_DIR, model_path)
        if not os.path.isfile(full):
            return {"ok": False, "error": f"Model file not found: {model_path}"}
        main_path = os.path.join(BACKEND_DIR, "main.py")
        try:
            with open(main_path, "r", encoding="utf-8") as f:
                content = f.read()
            import re
            new_content = re.sub(
                r'ObjectDetector\(model_name=["\']([^"\']+)["\']\)',
                f'ObjectDetector(model_name="{model_path}")',
                content,
            )
            if new_content == content:
                return {"ok": False, "error": "Could not find ObjectDetector line in main.py"}
            with open(main_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return {
                "ok": True,
                "message": f"Activated {model_path}. Restart backend to load the new model.",
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

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

        data_yaml_rel = config.get("data_yaml_path", "dataset/data.yaml")
        data_yaml_path = os.path.join(BACKEND_DIR, data_yaml_rel) if not os.path.isabs(data_yaml_rel) else data_yaml_rel
        if not os.path.isfile(data_yaml_path):
            return {"ok": False, "error": f"data.yaml not found: {data_yaml_rel}"}

        # Preflight validation: structure, paths, nc/names တွေ စစ်ပေးမယ်
        try:
            check = preflight_check_training(data_yaml_path)
            if not check.get("ok", False):
                errs = check.get("errors", [])
                return {
                    "ok": False,
                    "error": "Dataset preflight မအောင်မြင်ပါ။ " + " | ".join(errs),
                    "preflight": check,
                }
            # Warnings တွေရှိရင် logs ထဲမှာမှတ်ထားမယ်
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
                        self.logs.append(f"[VisionSync] ⚠️  Preflight warning: {w}")
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

                results = model.train(
                    data=yaml_path,
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    name=run_name,
                    plots=True,
                    project=os.path.join(BACKEND_DIR, "runs", "detect"),
                )

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
                sys.stdout = old_stdout
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
