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
    read_active_model,
)
from vlm_assistant import VLMAssistant

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

# Active model ßÇÇßÇ¡ßÇ» active_model.json ßÇÖßÇ╛ ßÇûßÇÉßÇ║ßÇ₧ßÇèßÇ║ (ßÇÖßÇ¢ßÇ╛ßÇ¡ßÇ¢ßÇäßÇ║ yolov8n.pt)ßüï
# ßÇñßÇößÇèßÇ║ßÇ╕ßÇûßÇ╝ßÇäßÇ╖ßÇ║ activate ßÇ£ßÇ»ßÇòßÇ║ßÇæßÇ¼ßÇ╕ßÇ₧ßÇ▒ßÇ¼ model ßÇ₧ßÇèßÇ║ server restart ßÇòßÇ╝ßÇ«ßÇ╕ßÇ£ßÇèßÇ║ßÇ╕ ßÇåßÇÇßÇ║ßÇíßÇ₧ßÇÇßÇ║ßÇ¥ßÇäßÇ║ßÇößÇ▒ßÇÖßÇèßÇ║ßüï
detector = ObjectDetector(model_name=read_active_model())
training_manager = TrainingManager()
vlm_assistant = VLMAssistant()

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
    conf: Optional[float] = Field(
        None,
        description="Confidence threshold (default 0.35). Custom model ßÇíßÇ₧ßÇàßÇ║ßÇÉßÇ╜ßÇ▒ßÇÇ confidence ßÇößÇ¡ßÇÖßÇ╖ßÇ║ßÇÉßÇÉßÇ║ßÇ£ßÇ¡ßÇ»ßÇ╖ 0.15 ßÇ£ßÇ▒ßÇ¼ßÇÇßÇ║ßÇößÇ▓ßÇ╖ ßÇàßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇèßÇ╖ßÇ║ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇ₧ßÇèßÇ║ßüï",
        ge=0.0,
        le=1.0,
    )


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


class ActivateModelRequest(BaseModel):
    model_path: str


class VLMDescribeRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image frame")


class VLMQueryRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image frame")
    question: str = Field(..., description="Natural language question about the image")


class AddClassRequest(BaseModel):
    yaml_path: str = Field(..., description="Master data.yaml ßÇûßÇ¡ßÇ»ßÇäßÇ║ßüÅ relative or absolute path (ßÇÑßÇòßÇÖßÇ¼ dataset/master/data.yaml)")
    class_name: str = Field(..., description="ßÇíßÇ₧ßÇàßÇ║ßÇæßÇèßÇ╖ßÇ║ßÇÖßÇèßÇ╖ßÇ║ Class ßÇößÇ¼ßÇÖßÇèßÇ║")


class MergeDatasetRequest(BaseModel):
    master_root: str = Field(..., description="Master dataset folder path (ßÇÑßÇòßÇÖßÇ¼ dataset/master)")
    source_root: str = Field(..., description="Dataset ßÇíßÇ₧ßÇàßÇ║ßÇ¢ßÇ╛ßÇ¡ßÇ₧ßÇ▒ßÇ¼ folder path (data.yaml ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ folder)")
    class_name: str = Field(..., description="Master ßÇæßÇ▓ßÇ₧ßÇ¡ßÇ»ßÇ╖ßÇæßÇèßÇ╖ßÇ║ßÇÖßÇèßÇ╖ßÇ║ / reuse ßÇÖßÇèßÇ╖ßÇ║ Class ßÇößÇ¼ßÇÖßÇèßÇ║")
    source_class_ids: Optional[List[int]] = Field(None, description="Source ßÇæßÇ▓ßÇÖßÇ╛ ßÇñ IDs ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ»ßÇ₧ßÇ¼ßÇÜßÇ░ßÇÖßÇèßÇ║ (None ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕)")
    merge_mode: str = Field(
        "auto",
        description="'auto' | 'collapse' (ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇÇßÇ¡ßÇ» class_name ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÉßÇèßÇ║ßÇ╕) | 'per_class' (source class ßÇößÇ¼ßÇÖßÇèßÇ║ßÇÉßÇàßÇ║ßÇüßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ╕ ßÇ₧ßÇ«ßÇ╕ßÇ₧ßÇößÇ╖ßÇ║ßÇæßÇèßÇ╖ßÇ║)",
    )


class ContinueFinetuneRequest(BaseModel):
    base_model: str = Field(..., description="ßÇÖßÇ░ßÇ£ ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ßÇòßÇ½ßÇ₧ßÇèßÇ╖ßÇ║ .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ£ßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕")
    source_root: str = Field(..., description="Dataset ßÇíßÇ₧ßÇàßÇ║ßÇ¢ßÇ╛ßÇ¡ßÇ₧ßÇ▒ßÇ¼ folder path")
    class_name: str = Field(..., description="ßÇòßÇàßÇ╣ßÇàßÇèßÇ║ßÇ╕ßÇíßÇ₧ßÇàßÇ║ßüÅ Class ßÇößÇ¼ßÇÖßÇèßÇ║")
    merge_mode: str = Field("auto", description="'auto' | 'collapse' | 'per_class'")
    epochs: int = 20
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.001
    freeze: int = Field(10, description="Backbone layer ßÇüßÇ▓ßÇæßÇ¼ßÇ╕ßÇÖßÇèßÇ╖ßÇ║ßÇíßÇ¢ßÇ▒ßÇíßÇÉßÇ╜ßÇÇßÇ║ (0 = ßÇÖßÇüßÇ▓)")
    run_name: str = "visionsync_master"


class DirectFinetuneRequest(BaseModel):
    base_model: str = Field(..., description="ßÇÖßÇ░ßÇ£ .pt (ßÇÑßÇòßÇÖßÇ¼ ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕) ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ£ßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕")
    run_name: str = "visionsync_master"
    epochs: int = 20
    imgsz: int = 640
    batch: int = 16
    lr0: float = 0.001
    freeze: int = Field(10, description="Backbone layer ßÇüßÇ▓ßÇæßÇ¼ßÇ╕ßÇÖßÇèßÇ╖ßÇ║ßÇíßÇ¢ßÇ▒ßÇíßÇÉßÇ╜ßÇÇßÇ║ (0 = ßÇÖßÇüßÇ▓)")


class ModelInfoRequest(BaseModel):
    pt_path: str = Field(..., description="Info ßÇÇßÇ¡ßÇ» ßÇ£ßÇ¡ßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ₧ßÇ▒ßÇ¼ .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ£ßÇÖßÇ║ßÇ╕ßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕")


class CocoReplayRequest(BaseModel):
    source: str = Field("val2017", description="'coco128' (ßüçMBßüè ßÇàßÇÖßÇ║ßÇ╕ßÇ¢ßÇößÇ║) ßÇ₧ßÇ¡ßÇ»ßÇ╖ 'val2017' (ßüêßüÇßüÇMBßüè ßÇÉßÇÇßÇÜßÇ║ßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇ¢ßÇößÇ║)")
    per_class: int = Field(30, ge=1, le=500, description="COCO class ßÇÉßÇàßÇ║ßÇüßÇ»ßÇ£ßÇ╗ßÇ╛ßÇäßÇ║ ßÇ¢ßÇèßÇ║ßÇÖßÇ╛ßÇößÇ║ßÇ╕ instance ßÇíßÇ¢ßÇ▒ßÇíßÇÉßÇ╜ßÇÇßÇ║")
    val_ratio: float = Field(0.2, ge=0.0, le=0.9)
    replace: bool = Field(True, description="ßÇíßÇ¢ßÇäßÇ║ßÇæßÇèßÇ╖ßÇ║ßÇæßÇ¼ßÇ╕ßÇ₧ßÇ▒ßÇ¼ replay ßÇÇßÇ¡ßÇ» ßÇûßÇ╗ßÇÇßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ ßÇíßÇ₧ßÇàßÇ║ßÇæßÇèßÇ╖ßÇ║ßÇÖßÇèßÇ║")


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

    _master_finetune_state["logs"].append("≡ƒÜÇ Continuous Fine-Tuning ßÇàßÇÉßÇäßÇ║ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║...")
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
            _on_log(f"Γ¥î {res['message']}")
    except Exception as e:
        _on_log(f"Γ¥î Unexpected error: {e}")
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
        # model_loaded=False ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║ detection ßÇÉßÇ╜ßÇ▒ßÇƒßÇ¼ random fake data ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï
        # load_error ßÇÇßÇ¡ßÇ» ßÇÇßÇ╝ßÇèßÇ╖ßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ ßÇíßÇÇßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ßÇ¢ßÇäßÇ║ßÇ╕ßÇÇßÇ¡ßÇ» ßÇ₧ßÇ¡ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇ₧ßÇèßÇ║ßüï
        "load_error": getattr(detector, "load_error", None),
        "nc": len(getattr(detector, "class_names", []) or []),
        "class_names": getattr(detector, "class_names", []) or [],
        "current_active_model": training_manager.current_model(),
        "timestamp": int(time.time()),
    }


@app.post("/detect", response_model=DetectResponse)
def detect_objects(req: DetectRequest):
    if not req.image:
        raise HTTPException(status_code=400, detail="Image base64 string is required")

    conf_th = 0.35 if req.conf is None else float(req.conf)
    mode_str = (req.mode or "general").lower()

    try:
        # If explicitly requested gemini/vlm mode, use Google Gemini Vision
        if mode_str in ("gemini", "vlm", "cloud"):
            detections = vlm_assistant.detect_objects(req.image, conf_threshold=conf_th)
            return {"detections": detections}

        detections = detector.detect(
            image_base64=req.image,
            mode=req.mode,
            conf_threshold=conf_th,
        )

        # If local model is fallback and Gemini is available, use Gemini Vision for accurate results
        if getattr(detector, "use_fallback", False) and vlm_assistant.is_configured:
            cloud_detections = vlm_assistant.detect_objects(req.image, conf_threshold=conf_th)
            if cloud_detections:
                return {"detections": cloud_detections}

        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/vlm/detect", response_model=DetectResponse)
def vlm_detect(req: DetectRequest):
    """
    Direct Gemini Vision Object Detection Endpoint.
    Detects objects and returns normalized bounding boxes using Google Gemini Vision.
    """
    if not req.image:
        raise HTTPException(status_code=400, detail="Image base64 string is required")
    conf_th = 0.35 if req.conf is None else float(req.conf)
    detections = vlm_assistant.detect_objects(req.image, conf_threshold=conf_th)
    return {"detections": detections}


@app.post("/vlm/describe")
def vlm_describe(req: VLMDescribeRequest):
    """
    Vision-Language Model (VLM) Scene Description Endpoint via Google Gemini Vision.
    Analyzes the image and returns a natural language description in Myanmar language.
    """
    return vlm_assistant.describe_scene(req.image)


@app.post("/vlm/query")
def vlm_query(req: VLMQueryRequest):
    """
    Vision-Language Model (VLM) Visual Question Answering (VQA) Endpoint via Google Gemini Vision.
    Answers any user question about the image in Myanmar language.
    """
    return vlm_assistant.ask_question(req.image, req.question)


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
            "message": "Training UI files not found. Visit /docs for the JSON API.",
            "apis": {
                "models": "/training/models",
                "activate_model": "POST /training/activate-model",
                "master_status": "/master/status",
                "master_info": "/master/info",
                "start_finetune": "POST /master/start-direct-finetune",
            }
        }
    return FileResponse(index_path, media_type="text/html")


@app.get("/training/models")
def list_models():
    return {
        "models": training_manager.list_models(),
        "active_model": training_manager.current_model(),
    }


@app.post("/training/activate-model")
def activate_model(req: ActivateModelRequest):
    result = training_manager.activate_model(req.model_path)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# /master/* ΓÇö Continuous Fine-Tuning & Master Dataset Management APIs
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
    return merge_dataset_into_master(m, s, req.class_name, req.source_class_ids, req.merge_mode)


@app.post("/master/upload-and-merge")
async def master_upload_and_merge(
    file: UploadFile = File(...),
    class_name: str = Form(...),
    master_root: str = Form("dataset/master"),
    dataset_name: Optional[str] = Form(None),
    source_class_ids: Optional[str] = Form(None),
    merge_mode: str = Form("auto"),
):
    """
    Upload ßÇ£ßÇ»ßÇòßÇ║ßÇ£ßÇ¡ßÇ»ßÇÇßÇ║ßÇÉßÇ▓ßÇ╖ zip file ßÇÇßÇ¡ßÇ» ßÇûßÇ▒ßÇ¼ßÇ║ßÇæßÇ»ßÇÉßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ ßÇÉßÇ¡ßÇ»ßÇÇßÇ║ßÇ¢ßÇ¡ßÇ»ßÇÇßÇ║ master dataset ßÇæßÇ▓ merge ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ▒ßÇ╕ßÇÉßÇ▓ßÇ╖
    ßÇ₧ßÇÇßÇ║ßÇÉßÇ▒ßÇ¼ßÇäßÇ╖ßÇ║ßÇ₧ßÇÇßÇ║ßÇ₧ßÇ¼ API ßÇÉßÇàßÇ║ßÇüßÇ»ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇÉßÇÜßÇ║ßüï
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    from master_builder import merge_dataset_into_master
    ids: Optional[List[int]] = None
    if source_class_ids:
        try:
            ids = [int(x.strip()) for x in source_class_ids.split(",") if x.strip()]
        except Exception:
            raise HTTPException(status_code=400, detail="source_class_ids ßÇÇßÇ¡ßÇ» 0,1,2 ßÇÆßÇ«ßÇòßÇ»ßÇ╢ßÇàßÇ╢ßÇößÇ▓ßÇ╖ ßÇòßÇ¡ßÇ»ßÇ╖ßÇòßÇ½")

    safe_name = dataset_name or (os.path.splitext(os.path.basename(file.filename))[0])
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in ("-", "_")) or f"src_{uuid.uuid4().hex[:8]}"

    os.makedirs(UPLOADS_DIR, exist_ok=True)
    zip_path = os.path.join(UPLOADS_DIR, f"{safe_name}_{uuid.uuid4().hex[:8]}.zip")
    try:
        with open(zip_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload save ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½: {e}")

    extract_dir = os.path.join(DATASET_DIR, safe_name)
    try:
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except Exception as e:
        try:
            shutil.rmtree(extract_dir, ignore_errors=True)
        finally:
            raise HTTPException(status_code=500, detail=f"Zip extract ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½: {e}")
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
            "message": f"Zip ßÇæßÇ▓ßÇÖßÇ╛ßÇ¼ data.yaml ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½ßüï Extract ßÇ£ßÇ»ßÇòßÇ║ßÇæßÇ¼ßÇ╕ßÇÉßÇ▓ßÇ╖ folder: {extract_dir}",
        }

    m = _resolve_rel(master_root)
    return merge_dataset_into_master(m, source_root, class_name, ids, merge_mode)


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


_coco_replay_state: Dict[str, Any] = {
    "status": "idle",
    "message": "",
    "progress_logs": [],
    "result": None,
    "started_at": None,
    "finished_at": None,
}


@app.get("/master/replay-status")
def master_replay_status():
    """COCO replay ßÇæßÇèßÇ╖ßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ßÇ£ßÇ¼ßÇ╕ / ßÇæßÇèßÇ╖ßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇ£ßÇ¼ßÇ╕ ßÇàßÇàßÇ║ßÇ¢ßÇößÇ║ßüï"""
    from coco_replay import replay_info
    from master_builder import MASTER_DIR
    state = dict(_coco_replay_state)
    state["info"] = replay_info(MASTER_DIR)
    return state


@app.post("/master/add-coco-replay")
def master_add_coco_replay(req: CocoReplayRequest):
    """
    ßÇÖßÇ░ßÇ£ COCO ßüêßüÇ ßÇÖßÇ╗ßÇ¡ßÇ»ßÇ╕ ßÇÖßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇàßÇ▒ßÇ¢ßÇößÇ║ COCO ßÇòßÇ»ßÇ╢ßÇíßÇüßÇ╗ßÇ¡ßÇ»ßÇ╖ßÇÇßÇ¡ßÇ» master dataset ßÇæßÇ▓ ßÇæßÇèßÇ╖ßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï

    Fine-tune ßÇÖßÇ£ßÇ»ßÇòßÇ║ßÇüßÇäßÇ║ ßÇÉßÇàßÇ║ßÇüßÇ½ßÇæßÇèßÇ╖ßÇ║ßÇæßÇ¼ßÇ╕ßÇ¢ßÇ»ßÇ╢ßÇûßÇ╝ßÇäßÇ╖ßÇ║ class ßÇíßÇ₧ßÇàßÇ║ßÇ¢ßÇ▒ßÇ¼ ßÇíßÇƒßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ßÇ¢ßÇ▒ßÇ¼
    ßÇÉßÇàßÇ║ßÇòßÇ╝ßÇ¡ßÇ»ßÇäßÇ║ßÇÉßÇèßÇ║ßÇ╕ ßÇ₧ßÇäßÇ║ßÇÜßÇ░ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇÖßÇèßÇ║ßüï Background ßÇÖßÇ╛ßÇ¼ run ßÇòßÇ╝ßÇ«ßÇ╕ /master/replay-status
    ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇàßÇàßÇ║ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇ₧ßÇèßÇ║ßüï
    """
    from coco_replay import SOURCES, add_coco_replay
    from master_builder import MASTER_DIR

    if req.source not in SOURCES:
        raise HTTPException(status_code=400,
                            detail=f"source ßÇÖßÇÖßÇ╛ßÇößÇ║ßÇòßÇ½: {req.source} (ßÇ¢ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇ₧ßÇèßÇ║: {list(SOURCES)})")
    if _coco_replay_state["status"] == "running":
        raise HTTPException(status_code=409, detail="COCO replay ßÇæßÇèßÇ╖ßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇÉßÇÜßÇ║ßüï")
    if _master_finetune_state["status"] == "running":
        raise HTTPException(status_code=409,
                            detail="Training ßÇ£ßÇ»ßÇòßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇÖßÇ╛ßÇ¼ dataset ßÇÇßÇ¡ßÇ» ßÇÖßÇòßÇ╝ßÇ▒ßÇ¼ßÇäßÇ║ßÇ╕ßÇ₧ßÇäßÇ╖ßÇ║ßÇòßÇ½ßüï ßÇòßÇ╝ßÇ«ßÇ╕ßÇÖßÇ╛ ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ½ßüï")

    def _runner() -> None:
        _coco_replay_state.update({
            "status": "running",
            "message": f"{req.source} ßÇÇßÇ¡ßÇ» ßÇæßÇèßÇ╖ßÇ║ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║...",
            "progress_logs": [],
            "result": None,
            "started_at": int(time.time() * 1000),
            "finished_at": None,
        })

        def _log(line: str) -> None:
            _coco_replay_state["progress_logs"].append(line)
            _coco_replay_state["message"] = line

        try:
            res = add_coco_replay(
                master_root=MASTER_DIR,
                source=req.source,
                per_class=req.per_class,
                val_ratio=req.val_ratio,
                replace=req.replace,
                on_log=_log,
            )
            _coco_replay_state.update({
                "status": "ok" if res.get("ok") else "error",
                "message": res.get("message", ""),
                "result": res,
                "finished_at": int(time.time() * 1000),
            })
        except Exception as e:
            _coco_replay_state.update({
                "status": "error",
                "message": f"Unexpected error: {e}",
                "finished_at": int(time.time() * 1000),
            })

    threading.Thread(target=_runner, daemon=True).start()
    return {
        "ok": True,
        "message": (
            f"COCO replay ({req.source}) ßÇæßÇèßÇ╖ßÇ║ßÇÉßÇ¼ßÇÇßÇ¡ßÇ» background ßÇÖßÇ╛ßÇ¼ ßÇàßÇÉßÇäßÇ║ßÇ£ßÇ¡ßÇ»ßÇÇßÇ║ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï "
            "/master/replay-status ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇàßÇàßÇ║ßÇåßÇ▒ßÇ╕ßÇòßÇ½ßüï"
        ),
        "approx_mb": SOURCES[req.source]["approx_mb"],
    }


@app.delete("/master/coco-replay")
def master_remove_coco_replay():
    """Master ßÇæßÇ▓ßÇÇ COCO replay ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇÇßÇ¡ßÇ» ßÇûßÇ╗ßÇÇßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï"""
    from coco_replay import remove_existing_replay
    from master_builder import MASTER_DIR
    if _master_finetune_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Training ßÇ£ßÇ»ßÇòßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇòßÇ½ßüï ßÇòßÇ╝ßÇ«ßÇ╕ßÇÖßÇ╛ ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ½ßüï")
    n = remove_existing_replay(MASTER_DIR)
    return {"ok": True, "removed_files": n, "message": f"replay ßÇûßÇ¡ßÇ»ßÇäßÇ║ {n} ßÇüßÇ» ßÇûßÇ╗ßÇÇßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï"}


@app.get("/master/audit")
def master_audit():
    """
    Master dataset ßÇæßÇ▓ class ßÇÉßÇàßÇ║ßÇüßÇ»ßÇüßÇ╗ßÇäßÇ║ßÇ╕ label ßÇÿßÇÜßÇ║ßÇößÇ╛ßÇàßÇ║ßÇüßÇ»ßÇ¢ßÇ╛ßÇ¡ßÇ£ßÇ▓ ßÇ¢ßÇ▒ßÇÉßÇ╜ßÇÇßÇ║ßÇòßÇ╝ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï
    Fine-tune ßÇÖßÇ£ßÇ»ßÇòßÇ║ßÇüßÇäßÇ║ "ßÇÿßÇÜßÇ║ class ßÇÉßÇ╜ßÇ▒ ßÇòßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇÖßÇ£ßÇ▓" ßÇÇßÇ¡ßÇ» ßÇÇßÇ╝ßÇ¡ßÇ»ßÇ₧ßÇ¡ßÇ¢ßÇößÇ║ßüï
    """
    from master_builder import MASTER_DIR, audit_master_coverage
    return audit_master_coverage(os.path.join(MASTER_DIR, "data.yaml"))


@app.get("/master/download-model")
def master_download_model(path: str):
    import urllib.parse
    decoded = urllib.parse.unquote(path)
    abs_path = _resolve_rel(decoded)
    if not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail=f".pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {decoded}")
    if not abs_path.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="download ßÇ£ßÇ»ßÇòßÇ║ßÇ£ßÇ¡ßÇ»ßÇ╖ßÇ¢ßÇÉßÇ¼ .pt ßÇûßÇ¡ßÇ»ßÇäßÇ║ßÇ₧ßÇ¼ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇÉßÇÜßÇ║")
    fname = os.path.basename(abs_path)
    return FileResponse(abs_path, filename=fname, media_type="application/octet-stream")


@app.post("/master/start-finetune")
def master_start_finetune(req: ContinueFinetuneRequest):
    """
    ßÇÉßÇàßÇ║ßÇüßÇ½ßÇÉßÇèßÇ║ßÇ╕: master dataset ßÇíßÇ£ßÇ¡ßÇ»ßÇíßÇ£ßÇ╗ßÇ▒ßÇ¼ßÇÇßÇ║ßÇÉßÇèßÇ║ßÇåßÇ▒ßÇ¼ßÇÇßÇ║ ΓåÆ class ßÇæßÇèßÇ╖ßÇ║ ΓåÆ dataset merge
    ΓåÆ base .pt ßÇÖßÇ╛ fine-tune ßÇòßÇ╝ßÇ»ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ .pt ßÇíßÇ₧ßÇàßÇ║ßÇÉßÇàßÇ║ßÇüßÇ»ßÇÉßÇèßÇ║ßÇ╕ (class ßüêßüÇ+ßÇíßÇ₧ßÇàßÇ║) ßÇæßÇ╜ßÇÇßÇ║ßÇ£ßÇ¼ßÇàßÇ▒ßÇ₧ßÇèßÇ║ßüï
    """
    from master_builder import (
        MASTER_DIR,
        _ensure_master_structure,
        extract_model_info,
        merge_dataset_into_master,
        continuous_finetune,
    )

    if _master_finetune_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Continuous Fine-Tuning ßÇ£ßÇ»ßÇòßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇÉßÇÜßÇ║ßüï ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ╜ßÇ¼ßÇ╕ßÇÖßÇ╛ ßÇæßÇòßÇ║ßÇ£ßÇ»ßÇòßÇ║ßÇòßÇ½ßüï")

    base_pt = _resolve_rel(req.base_model)
    source_root = _resolve_rel(req.source_root)
    if not os.path.isfile(base_pt):
        raise HTTPException(status_code=400, detail=f"Base .pt ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {base_pt}")
    if not os.path.isdir(source_root):
        raise HTTPException(status_code=400, detail=f"Source folder ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {source_root}")

    info = extract_model_info(base_pt)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail=f"Base model info ßÇÖßÇ¢ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇòßÇ½: {info.get('message')}")

    _ensure_master_structure(MASTER_DIR)
    master_yaml = os.path.join(MASTER_DIR, "data.yaml")

    merge_res = merge_dataset_into_master(MASTER_DIR, source_root, req.class_name, None, req.merge_mode)
    if not merge_res["ok"]:
        raise HTTPException(status_code=500, detail=f"Dataset merge ßÇÖßÇíßÇ▒ßÇ¼ßÇäßÇ║ßÇÖßÇ╝ßÇäßÇ║ßÇòßÇ½: {merge_res['message']}")

    def _runner() -> None:
        _master_finetune_state.update({
            "status": "running",
            "message": f"{merge_res['message']} | Fine-tuning ßÇàßÇÉßÇäßÇ║ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║...",
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
            freeze=req.freeze,
            run_name=req.run_name,
            **cb,
        ))

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    return {
        "ok": True,
        "message": "Continuous Fine-Tuning ßÇíßÇ£ßÇ»ßÇòßÇ║ßÇÇßÇ¡ßÇ» background ßÇÖßÇ╛ßÇ¼ ßÇàßÇÉßÇäßÇ║ßÇ£ßÇ¡ßÇ»ßÇÇßÇ║ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï /master/status ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇàßÇàßÇ║ßÇåßÇ▒ßÇ╕ßÇòßÇ½ßüï",
        "base_nc": info.get("nc"),
        "merge": merge_res,
    }


@app.post("/master/start-direct-finetune")
def master_start_direct_finetune(req: DirectFinetuneRequest):
    """
    Master dataset ßÇÇßÇ¡ßÇ» ßÇíßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇòßÇ╝ßÇ»ßÇòßÇ╝ßÇ«ßÇ╕ (ßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇ₧ßÇòßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ Merges ßÇÖßÇ╗ßÇ¼ßÇ╕ßÇíßÇ¼ßÇ╕ßÇ£ßÇ»ßÇ╢ßÇ╕ßÇòßÇ½ßÇ¥ßÇäßÇ║ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ╝ßÇ«)
    base .pt ßÇûßÇ╝ßÇäßÇ╖ßÇ║ Continuous Fine-Tune ßÇÇßÇ¡ßÇ» ßÇÉßÇ¡ßÇ»ßÇÇßÇ║ßÇ¢ßÇ¡ßÇ»ßÇÇßÇ║ ßÇàßÇÉßÇäßÇ║ßÇòßÇ▒ßÇ╕ßÇ₧ßÇèßÇ║ßüï
    ßÇíßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇòßÇ╝ßÇ»ßÇ₧ßÇ░ßÇ₧ßÇèßÇ║ ßÇíßÇ¢ßÇäßÇ║ßÇÇ zip upload+merge ßÇòßÇ╝ßÇ«ßÇ╕ßÇòßÇ╝ßÇ«ßÇåßÇ¡ßÇ»ßÇ¢ßÇäßÇ║ ßÇÆßÇ« endpoint ßÇÇßÇ¡ßÇ» ßÇ₧ßÇ»ßÇ╢ßÇ╕ßÇòßÇ½ßüï
    """
    from master_builder import (
        MASTER_DIR,
        _ensure_master_structure,
        extract_model_info,
        continuous_finetune,
    )
    import os as _os

    if _master_finetune_state["status"] == "running":
        raise HTTPException(status_code=409, detail="Continuous Fine-Tuning ßÇ£ßÇ»ßÇòßÇ║ßÇößÇ▒ßÇåßÇ▓ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇÉßÇÜßÇ║ßüï")

    base_pt = _resolve_rel(req.base_model)
    if not _os.path.isfile(base_pt):
        raise HTTPException(status_code=400, detail=f"Base .pt ßÇÖßÇÉßÇ╜ßÇ▒ßÇ╖ßÇ¢ßÇ╛ßÇ¡ßÇòßÇ½: {base_pt}")

    info = extract_model_info(base_pt)
    if not info.get("ok"):
        raise HTTPException(status_code=400, detail=f"Base model info ßÇÖßÇ¢ßÇößÇ¡ßÇ»ßÇäßÇ║ßÇòßÇ½: {info.get('message')}")

    _ensure_master_structure(MASTER_DIR)
    master_yaml = _os.path.join(MASTER_DIR, "data.yaml")

    def _runner() -> None:
        _master_finetune_state.update({
            "status": "running",
            "message": "Dataset ßÇÖßÇ╗ßÇ¼ßÇ╕ merge ßÇòßÇ╝ßÇ«ßÇ╕ßÇ₧ßÇ¼ßÇ╕ßÇûßÇ╝ßÇàßÇ║ßÇòßÇ½ßÇ₧ßÇèßÇ║ßüï Fine-tuning ßÇàßÇÉßÇäßÇ║ßÇößÇ▒ßÇòßÇ½ßÇ₧ßÇèßÇ║...",
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
            freeze=req.freeze,
            run_name=req.run_name,
            **cb,
        ))

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    return {
        "ok": True,
        "message": "Continuous Fine-Tuning ßÇíßÇ£ßÇ»ßÇòßÇ║ßÇÇßÇ¡ßÇ» background ßÇÖßÇ╛ßÇ¼ ßÇàßÇÉßÇäßÇ║ßÇ£ßÇ¡ßÇ»ßÇÇßÇ║ßÇòßÇ½ßÇòßÇ╝ßÇ«ßüï /master/status ßÇûßÇ╝ßÇäßÇ╖ßÇ║ ßÇàßÇàßÇ║ßÇåßÇ▒ßÇ╕ßÇòßÇ½ßüï",
        "base_nc": info.get("nc"),
        "master_yaml": master_yaml,
    }


if __name__ == "__main__":
    ip = get_local_ip()
    print("\n" + "=" * 70)
    print("≡ƒÜÇ VisionSync AI Backend Server Started!")
    print(f"≡ƒô▒ Connect from Mobile Phone: http://{ip}:8000")
    print(f"≡ƒîÉ Localhost:               http://localhost:8000")
    print(f"≡ƒôä Swagger API Docs:        http://{ip}:8000/docs")
    print(f"≡ƒºá Training Dashboard:      http://{ip}:8000/training")
    print("=" * 70 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
