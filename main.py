import json
import os
import shutil
import socket
import tempfile
import threading
import time
import uuid
import zipfile
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from detector import ObjectDetector
from training_manager import (
    BACKEND_DIR,
    UPLOADS_DIR,
    DATASET_DIR,
    TrainingManager,
)

app = FastAPI(
    title="VisionSync AI Backend Server",
    description="FastAPI + Ultralytics YOLOv8 backend for real-time object detection and accessibility assistance, with custom model training dashboard.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TRAINING_UI_DIR = os.path.join(BACKEND_DIR, "training_ui")
os.makedirs(TRAINING_UI_DIR, exist_ok=True)
if os.path.isdir(TRAINING_UI_DIR):
    app.mount("/training/assets", StaticFiles(directory=TRAINING_UI_DIR), name="training_assets")

detector = ObjectDetector(model_name="yolov8n.pt")
training_manager = TrainingManager()

REPORTS_FILE = os.path.join(BACKEND_DIR, "reports_log.json")


def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded JPEG or PNG image frame")
    mode: str = Field("general", description="Detection mode: 'general' | 'security' | 'industrial'")


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionItem(BaseModel):
    classId: str
    confidence: float
    box: BoundingBox


class DetectResponse(BaseModel):
    detections: List[DetectionItem]


class ReportRequest(BaseModel):
    id: str
    historyEntryId: str
    classId: str
    reason: str
    note: Optional[str] = ""
    timestamp: Optional[int] = None


class TrainStartRequest(BaseModel):
    data_yaml_path: str = Field(..., description="Relative path to dataset data.yaml (e.g. dataset/my_dataset/data.yaml)")
    base_model: str = "yolov8n.pt"
    epochs: int = 50
    img_size: int = 640
    batch_size: int = 16
    run_name: str = "visionsync_custom"


class ActivateModelRequest(BaseModel):
    model_path: str


class AddClassRequest(BaseModel):
    yaml_path: str = Field(..., description="Master data.yaml ဖိုင်၏ relative or absolute path (ဥပမာ dataset/master/data.yaml)")
    class_name: str = Field(..., description="အသစ်ထည့်မည့် Class နာမည်")


class MergeDatasetRequest(BaseModel):
    master_root: str = Field(..., description="Master dataset folder path (ဥပမာ dataset/master)")
    source_root: str = Field(..., description="Dataset အသစ်ရှိသော folder path (data.yaml ပါသည့် folder)")
    class_name: str = Field(..., description="Master ထဲသို့ထည့်မည့် / reuse မည့် Class နာမည်")
    source_class_ids: Optional[List[int]] = Field(None, description="Source ထဲမှ ဤ IDs များကိုသာယူမည် (None ဆိုရင်အားလုံး)")


class ContinueFinetuneRequest(BaseModel):
    base_model: str = Field(..., description="မူလ ၈၀ မျိုးပါသည့် .pt ဖိုင်လမ်းကြောင်း")
    source_root: str = Field(..., description="Dataset အသစ်ရှိသော folder path")
    class_name: str = Field(..., description="ပစ္စည်းအသစ်၏ Class နာမည်")
    epochs: int = 20
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.001
    run_name: str = "visionsync_master"


class DirectFinetuneRequest(BaseModel):
    base_model: str = Field(..., description="မူလ .pt (ဥပမာ ၈၀ မျိုး) ဖိုင်လမ်းကြောင်း")
    run_name: str = "visionsync_master"
    epochs: int = 20
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.001


class ModelInfoRequest(BaseModel):
    pt_path: str = Field(..., description="Info ကို လိုချင်သော .pt ဖိုင်လမ်းကြောင်း")


_master_finetune_state: Dict[str, Any] = {
    "status": "idle",
    "message": "",
    "best_pt": None,
    "archived_pt": None,
    "total_nc": 0,
    "names": [],
    "progress": 0,
    "logs": [],
    "current_epoch": 0,
    "total_epochs": 0,
    "started_at": None,
    "finished_at": None,
}


def _master_finetune_run(continuous_finetune_fn: Any) -> None:
    """Run continuous_finetune in background with live progress + logs wired to state."""
    def _on_progress(epoch: int, total: int) -> None:
        _master_finetune_state["current_epoch"] = epoch
        _master_finetune_state["total_epochs"] = total
        _master_finetune_state["progress"] = int(min(100, (epoch / max(1, total)) * 100))

    def _on_log(line: str) -> None:
        _master_finetune_state["logs"].append(line)

    _master_finetune_state["logs"].append("🚀 Continuous Fine-Tuning စတင်နေပါသည်...")
    try:
        res = continuous_finetune_fn(on_log=_on_log, on_progress=_on_progress)
        _master_finetune_state.update({
            "status": "ok" if res["ok"] else "error",
            "message": res["message"],
            "best_pt": res.get("best_pt"),
            "archived_pt": res.get("archived_pt"),
            "total_nc": res.get("total_nc", 0),
            "names": res.get("names", []),
            "finished_at": int(time.time() * 1000),
        })
        if not res.get("ok"):
            _on_log(f"❌ {res['message']}")
    except Exception as e:
        _on_log(f"❌ Unexpected error: {e}")
        _master_finetune_state.update({
            "status": "error",
            "message": f"Unexpected error: {e}",
            "finished_at": int(time.time() * 1000),
        })


@app.get("/")
def root():
    local_ip = get_local_ip()
    return {
        "name": "VisionSync AI Backend Server",
        "status": "running",
        "local_ip": local_ip,
        "connect_url": f"http://{local_ip}:8000",
        "docs_url": f"http://{local_ip}:8000/docs",
        "health_check": f"http://{local_ip}:8000/health",
        "training_dashboard": f"http://{local_ip}:8000/training",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": not detector.use_fallback,
        "model_name": detector.model_name,
        "current_active_model": training_manager.current_model(),
        "timestamp": int(time.time()),
    }


@app.post("/detect", response_model=DetectResponse)
def detect_objects(req: DetectRequest):
    if not req.image:
        raise HTTPException(status_code=400, detail="Image base64 string is required")

    try:
        detections = detector.detect(
            image_base64=req.image,
            mode=req.mode,
            conf_threshold=0.35,
        )
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/reports")
def submit_report(report: ReportRequest):
    reports = []
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                reports = json.load(f)
        except Exception:
            reports = []

    report_data = report.dict()
    if not report_data.get("timestamp"):
        report_data["timestamp"] = int(time.time() * 1000)

    reports.append(report_data)

    with open(REPORTS_FILE, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)

    return {"status": "success", "message": "Report logged for dataset retraining"}


@app.get("/reports")
def get_reports():
    if os.path.exists(REPORTS_FILE):
        try:
            with open(REPORTS_FILE, "r", encoding="utf-8") as f:
                return {"reports": json.load(f)}
        except Exception:
            return {"reports": []}
    return {"reports": []}


@app.get("/training")
def training_dashboard():
    index_path = os.path.join(TRAINING_UI_DIR, "index.html")
    if not os.path.isfile(index_path):
        return {
            "message": "Training UI files not found. Visit /training/status for the JSON API.",
            "apis": {
                "status": "/training/status",
                "start": "POST /training/start",
                "datasets": "/training/datasets",
                "models": "/training/models",
                "upload_dataset": "POST /training/upload-dataset",
                "activate_model": "POST /training/activate-model",
            }
        }
    return FileResponse(index_path, media_type="text/html")


@app.get("/training/status")
def training_status():
    return training_manager.get_state()


@app.get("/training/logs")
def training_logs(since: Optional[int] = 0):
    state = training_manager.get_state()
    logs = state["logs"]
    start_idx = max(0, int(since or 0))
    return {
        "status": state["status"],
        "progress": state["progress"],
        "total_lines": len(logs),
        "new_since": len(logs),
        "logs": logs[start_idx:],
        "error": state["error"],
        "result_weights": state["result_weights"],
    }


@app.get("/training/datasets")
def list_datasets():
    return {"datasets": training_manager.list_datasets()}


@app.get("/training/models")
def list_models():
    return {
        "models": training_manager.list_models(),
        "active_model": training_manager.current_model(),
    }


@app.post("/training/start")
def start_training(req: TrainStartRequest):
    result = training_manager.start_training({
        "data_yaml_path": req.data_yaml_path,
        "base_model": req.base_model,
        "epochs": req.epochs,
        "img_size": req.img_size,
        "batch_size": req.batch_size,
        "run_name": req.run_name,
    })
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/training/stop")
def stop_training():
    result = training_manager.stop_training()
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/training/activate-model")
def activate_model(req: ActivateModelRequest):
    result = training_manager.activate_model(req.model_path)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/training/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    safe_name = dataset_name or (os.path.splitext(os.path.basename(file.filename))[0])
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "_")) or f"dataset_{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(UPLOADS_DIR, f"{safe_name}_{uuid.uuid4().hex[:8]}.zip")
    try:
        with open(save_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    extract_result = training_manager.extract_uploaded_zip(save_path, safe_name)
    try:
        os.remove(save_path)
    except Exception:
        pass

    if not extract_result["ok"]:
        raise HTTPException(status_code=500, detail=f"Extract failed: {extract_result.get('error')}")

    ds_list = training_manager.list_datasets()
    matched = next((d for d in ds_list if d["name"] == safe_name), None)
    return {
        "ok": True,
        "dataset_name": safe_name,
        "dataset": matched,
        "hint": "If the dataset is nested inside a subfolder, adjust the data.yaml path accordingly, or move it under dataset/<name>/ directly.",
    }


# ---------------------------------------------------------------------------
# /master/* — Continuous Fine-Tuning & Master Dataset Management APIs
# ---------------------------------------------------------------------------

def _resolve_rel(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(BACKEND_DIR, path))


@app.post("/master/add-class")
def master_add_class(req: AddClassRequest):
    from master_builder import add_class_to_yaml
    yaml_path = _resolve_rel(req.yaml_path)
    return add_class_to_yaml(yaml_path, req.class_name)


@app.post("/master/merge-dataset")
def master_merge_dataset(req: MergeDatasetRequest):
    from master_builder import merge_dataset_into_master
    m = _resolve_rel(req.master_root)
    s = _resolve_rel(req.source_root)
    return merge_dataset_into_master(m, s, req.class_name, req.source_class_ids)


@app.post("/master/upload-and-merge")
async def master_upload_and_merge(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    master_root: str = Form("dataset/master"),
    dataset_name: Optional[str] = Form(None),
    source_class_ids: Optional[str] = Form(None),
):
    """
    Upload လုပ်လိုက်တဲ့ zip file ကို ဖော်ထုတ်ပြီး တိုက်ရိုက် master dataset ထဲ merge လုပ်ပေးတဲ့
    သက်တောင့်သက်သာ API တစ်ခုဖြစ်ပါတယ်။
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    from master_builder import merge_dataset_into_master
    ids: Optional[List[int]] = None
    if source_class_ids:
        try:
            ids = [int(x.strip()) for x in source_class_ids.split(",") if x.strip()]
        except Exception:
            raise HTTPException(status_code=400, detail="source_class_ids ကို 0,1,2 ဒီပုံစံနဲ့ ပို့ပါ")

    safe_name = dataset_name or (os.path.splitext(os.path.basename(file.filename))[0])
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "_")) or f"src_{uuid.uuid4().hex[:8]}"

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    zip_path = os.path.join(UPLOADS_DIR, f"{safe_name}_{uuid.uuid4().hex[:8]}.zip")
    try:
        with open(zip_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload save မအောင်မြင်ပါ: {e}")

    extract_dir = os.path.join(DATASET_DIR, safe_name)
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        try:
            shutil.rmtree(extract_dir, ignore_errors=True)
        finally:
            raise HTTPException(status_code=500, detail=f"Zip extract မအောင်မြင်ပါ: {e}")
    finally:
        try:
            os.remove(zip_path)
        except Exception:
            pass

    data_yaml_candidates = [
        os.path.join(extract_dir, "data.yaml"),
        os.path.join(extract_dir, "data.yml"),
    ]
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.lower() in ("data.yaml", "data.yml"):
                data_yaml_candidates.append(os.path.join(root, f))
    source_root = None
    for cand in data_yaml_candidates:
        if os.path.isfile(cand):
            source_root = os.path.dirname(cand)
            break
    if not source_root:
        return {
            "ok": False,
            "message": f"Zip ထဲမှာ data.yaml မတွေ့ရှိပါ။ Extract လုပ်ထားတဲ့ folder: {extract_dir}",
        }

    m = _resolve_rel(master_root)
    return merge_dataset_into_master(m, source_root, class_name, ids)


@app.post("/master/model-info")
def master_model_info(req: ModelInfoRequest):
    from master_builder import extract_model_info
    pt = _resolve_rel(req.pt_path)
    return extract_model_info(pt)


@app.get("/master/status")
def master_get_status():
    return dict(_master_finetune_state)


@app.get("/master/info")
def master_get_info():
    from master_builder import MASTER_DIR, _ensure_master_structure, _load_yaml, _normalize_names

    _ensure_master_structure(MASTER_DIR)
    yaml_path = os.path.join(MASTER_DIR, "data.yaml")
    info: Dict[str, Any] = {
        "master_dir": MASTER_DIR,
        "yaml_path": yaml_path,
        "exists": os.path.isdir(MASTER_DIR),
        "nc": 0,
        "names": [],
        "images_train": 0,
        "images_val": 0,
    }
    try:
        data = _load_yaml(yaml_path)
        names = _normalize_names(data.get("names", {}))
        info["nc"] = int(data.get("nc", len(names)))
        sorted_ids = sorted(names.keys())
        info["names"] = [{"id": i, "name": names[i]} for i in sorted_ids]
    except Exception as e:
        info["yaml_error"] = str(e)

    def count_img(d: str) -> int:
        p = os.path.join(MASTER_DIR, "images", d)
        if not os.path.isdir(p):
            return 0
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        try:
            return sum(1 for f in os.listdir(p)
                       if os.path.isfile(os.path.join(p, f)) and os.path.splitext(f)[1].lower() in exts)
        except Exception:
            return 0

    info["images_train"] = count_img("train")
    info["images_val"] = count_img("val")
    return info


@app.get("/master/download-model")
def master_download_model(path: str):
    import urllib.parse
    decoded = urllib.parse.unquote(path)
    abs_path = _resolve_rel(decoded)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f".pt ဖိုင်မတွေ့ရှိပါ: {decoded}")
    if not abs_path.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="download လုပ်လို့ရတာ .pt ဖိုင်သာဖြစ်ပါတယ်")
    fname = os.path.basename(abs_path)
    return FileResponse(abs_path, filename=fname, media_type="application/octet-stream")


@app.post("/master/start-finetune")
def master_start_finetune(req: ContinueFinetuneRequest):
    """
    တစ်ခါတည်း: master dataset အလိုအလျောက်တည်ဆောက် → class ထည့် → dataset merge
    → base .pt မှ fine-tune ပြုလုပ်ပြီး .pt အသစ်တစ်ခုတည်း (class ၈၀+အသစ်) ထွက်လာစေသည်။
    """
    from master_builder import (
        MASTER_DIR,
        _ensure_master_structure,
        extract_model_info,
        merge_dataset_into_master,
        continuous_finetune,
    )

    if _master_finetune_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Continuous Fine-Tuning လုပ်နေဆဲဖြစ်ပါတယ်။ ပြီးသွားမှ ထပ်လုပ်ပါ။")

    base_pt = _resolve_rel(req.base_model)
    source_root = _resolve_rel(req.source_root)
    if not os.path.isfile(base_pt):
        raise HTTPException(status_code=400, detail=f"Base .pt မတွေ့ရှိပါ: {base_pt}")
    if not os.path.isdir(source_root):
        raise HTTPException(status_code=400, detail=f"Source folder မတွေ့ရှိပါ: {source_root}")

    info = extract_model_info(base_pt)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail=f"Base model info မရနိုင်ပါ: {info.get('message')}")

    _ensure_master_structure(MASTER_DIR)
    master_yaml = os.path.join(MASTER_DIR, "data.yaml")

    merge_res = merge_dataset_into_master(MASTER_DIR, source_root, req.class_name)
    if not merge_res["ok"]:
        raise HTTPException(status_code=500, detail=f"Dataset merge မအောင်မြင်ပါ: {merge_res['message']}")

    def _runner() -> None:
        _master_finetune_state.update({
            "status": "running",
            "message": f"{merge_res['message']} | Fine-tuning စတင်နေပါသည်...",
            "best_pt": None,
            "archived_pt": None,
            "total_nc": info.get("nc", 0),
            "names": [],
            "progress": 0,
            "logs": [],
            "current_epoch": 0,
            "total_epochs": int(req.epochs),
            "started_at": int(time.time() * 1000),
            "finished_at": None,
        })
        _master_finetune_run(lambda **cb: continuous_finetune(
            base_model_path=base_pt,
            data_yaml_path=master_yaml,
            epochs=req.epochs,
            imgsz=req.imgsz,
            batch=req.batch,
            lr0=req.lr0,
            run_name=req.run_name,
            **cb,
        ))

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    return {
        "ok": True,
        "message": "Continuous Fine-Tuning အလုပ်ကို background မှာ စတင်လိုက်ပါပြီ။ /master/status ဖြင့် စစ်ဆေးပါ။",
        "base_nc": info.get("nc"),
        "merge": merge_res,
    }


@app.post("/master/start-direct-finetune")
def master_start_direct_finetune(req: DirectFinetuneRequest):
    """
    Master dataset ကို အသုံးပြုပြီး (သုံးသပ်ပြီးသား Merges များအားလုံးပါဝင်ပြီးပြီ)
    base .pt ဖြင့် Continuous Fine-Tune ကို တိုက်ရိုက် စတင်ပေးသည်။
    အသုံးပြုသူသည် အရင်က zip upload+merge ပြီးပြီဆိုရင် ဒီ endpoint ကို သုံးပါ။
    """
    from master_builder import (
        MASTER_DIR,
        _ensure_master_structure,
        extract_model_info,
        continuous_finetune,
    )
    import os as _os

    if _master_finetune_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Continuous Fine-Tuning လုပ်နေဆဲဖြစ်ပါတယ်။")

    base_pt = _resolve_rel(req.base_model)
    if not _os.path.isfile(base_pt):
        raise HTTPException(status_code=400, detail=f"Base .pt မတွေ့ရှိပါ: {base_pt}")

    info = extract_model_info(base_pt)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail=f"Base model info မရနိုင်ပါ: {info.get('message')}")

    _ensure_master_structure(MASTER_DIR)
    master_yaml = _os.path.join(MASTER_DIR, "data.yaml")

    def _runner() -> None:
        _master_finetune_state.update({
            "status": "running",
            "message": "Dataset များ merge ပြီးသားဖြစ်ပါသည်။ Fine-tuning စတင်နေပါသည်...",
            "best_pt": None,
            "archived_pt": None,
            "total_nc": info.get("nc", 0),
            "names": [],
            "progress": 0,
            "logs": [],
            "current_epoch": 0,
            "total_epochs": int(req.epochs),
            "started_at": int(time.time() * 1000),
            "finished_at": None,
        })
        _master_finetune_run(lambda **cb: continuous_finetune(
            base_model_path=base_pt,
            data_yaml_path=master_yaml,
            epochs=req.epochs,
            imgsz=req.imgsz,
            batch=req.batch,
            lr0=req.lr0,
            run_name=req.run_name,
            **cb,
        ))

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    return {
        "ok": True,
        "message": "Continuous Fine-Tuning အလုပ်ကို background မှာ စတင်လိုက်ပါပြီ။ /master/status ဖြင့် စစ်ဆေးပါ။",
        "base_nc": info.get("nc"),
        "master_yaml": master_yaml,
    }


if __name__ == "__main__":
    ip = get_local_ip()
    print("\n" + "=" * 70)
    print("🚀 VisionSync AI Backend Server Started!")
    print(f"📱 Connect from Mobile Phone: http://{ip}:8000")
    print(f"🌐 Localhost:               http://localhost:8000")
    print(f"📄 Swagger API Docs:        http://{ip}:8000/docs")
    print(f"🧠 Training Dashboard:      http://{ip}:8000/training")
    print("=" * 70 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
