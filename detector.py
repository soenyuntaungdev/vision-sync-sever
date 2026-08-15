import base64
import io
import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np
from PIL import Image

logger = logging.getLogger("visionsync")
logging.basicConfig(level=logging.INFO)

# Maximum image dimension for YOLO inference. YOLO natively processes at 640px;
# passing larger frames wastes compute without accuracy gain.
MAX_INFER_DIM = 640

# COCO 80 class names. MODE_CLASSES filters are based on these names.
# Any class not in this set is treated as a custom class and always allowed
# through the mode filter so fine-tuned classes are never hidden.
COCO_CLASS_NAMES: Set[str] = {
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
}

# Mode filtering definitions matching constants/modes.ts in VisionSync app.
# Each set includes both COCO-style names AND LVIS-style names (underscore/parentheses
# variants) so that filtering works regardless of the loaded model naming convention.
MODE_CLASSES: Dict[str, Optional[Set[str]]] = {
    "general": None,  # All classes (no filter)
    "security": {
        # People & personal items
        "person",
        "backpack",
        "handbag", "handbag_(purse)",
        "suitcase", "suitcase_(luggage)",
        # Weapons / tools
        "knife", "knife_(kitchen_utensil)",
        "scissors",
        # Electronics
        "cell phone", "cellular_telephone", "mobile_phone",
        # Vehicles
        "car", "car_(automobile)",
        "motorcycle", "motorbike",
        "bicycle",
        "truck", "truck_(vehicle)",
        "bus", "bus_(vehicle)",
    },
    "industrial": {
        # Heavy vehicles
        "truck", "truck_(vehicle)",
        "bus", "bus_(vehicle)",
        "train",
        # Furniture / fixtures
        "bench",
        "chair",
        # Tools
        "scissors",
        "knife", "knife_(kitchen_utensil)",
        # Appliances
        "oven", "oven_(kitchen_appliance)",
        "microwave", "microwave_oven",
        "toaster",
        "refrigerator",
        # Electronics / equipment
        "laptop", "laptop_computer",
        "keyboard", "computer_keyboard",
        "bottle",
        "fire hydrant", "fire_hydrant",
    },
}


class ObjectDetector:
    """
    Ultralytics YOLO object detector supporting COCO (80 classes) and Open Images
    v7 (601 classes) pretrained weights. Falls back to mock/synthetic detections
    automatically if model weights are unavailable or PyTorch/Ultralytics is absent.
    """

    def __init__(self, model_name: str = "yolov8n-oiv7.pt", allow_fallback: bool = True):
        self.model_name = model_name
        self.model = None
        self.use_fallback = False
        self.load_error: Optional[str] = None
        self.class_names: List[str] = []
        # allow_fallback=False raises on load failure instead of silently producing
        # fake detections that mislead the mobile app into thinking the model works.
        self.allow_fallback = allow_fallback
        self._load_model()

    def _load_model(self):
        self.load_error = None
        self.class_names = []
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            names_attr = getattr(getattr(self.model, "model", None), "names", None)
            if names_attr is None:
                names_attr = getattr(self.model, "names", None)
            if isinstance(names_attr, dict):
                self.class_names = [str(names_attr[k]) for k in sorted(names_attr.keys())]
            elif isinstance(names_attr, (list, tuple)):
                self.class_names = [str(n) for n in names_attr]
            logger.info(
                f"YOLO model loaded successfully! nc={len(self.class_names)} "
                f"classes={self.class_names[:5]}{'...' if len(self.class_names) > 5 else ''}"
            )
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            if not self.allow_fallback:
                logger.error(f"Model load failed for '{self.model_name}': {self.load_error}")
                raise
            logger.warning(f"Could not load Ultralytics YOLO model ({e}). Using smart fallback detector.")
            self.use_fallback = True

    def _effective_allowed(self, mode: str, model_names: Optional[Set[str]]) -> Optional[Set[str]]:
        """
        Computes the effective allowed class set for a given mode, taking into account
        the model class list so that custom (non-COCO) classes are always permitted.

        Returns None when no filter should be applied (mode=general, or no overlap).
        """
        base = MODE_CLASSES.get(mode, None)
        if base is None:
            return None
        if not model_names:
            return base
        custom_names = model_names - COCO_CLASS_NAMES
        allowed = set(base) | custom_names
        if not (model_names & allowed):
            # Model has no classes matching this mode -> show all rather than nothing
            return None
        return allowed

    def decode_base64_image(self, base64_str: str) -> Optional[Image.Image]:
        """
        Decodes a base64 encoded image string (with or without data URI prefix) into a PIL Image.
        """
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            image_bytes = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None

    def detect(self, image_base64: str, mode: str = "general", conf_threshold: float = 0.35) -> List[Dict[str, Any]]:
        """
        Runs object detection on the base64 image and returns normalized BoundingBoxes
        matching the VisionSync frontend schema:
        [
          {
            "classId": "chair",
            "confidence": 0.88,
            "box": { "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4 }
          }
        ]
        """
        image = self.decode_base64_image(image_base64)
        if image is None:
            return []

        img_w, img_h = image.size
        if img_w <= 0 or img_h <= 0:
            return []

        allowed_classes = self._effective_allowed(mode, set(self.class_names) if self.class_names else None)

        if not self.use_fallback and self.model is not None:
            try:
                if max(image.size) > MAX_INFER_DIM:
                    ratio = MAX_INFER_DIM / max(image.size)
                    image = image.resize(
                        (round(image.width * ratio), round(image.height * ratio)),
                        Image.LANCZOS,
                    )
                img_w, img_h = image.size
                img_np = np.array(image)
                results = self.model.predict(
                    img_np,
                    conf=conf_threshold,
                    imgsz=640,   # native YOLO resolution -- fastest inference
                    verbose=False,
                )

                detections = []
                for result in results:
                    boxes = result.boxes
                    # Re-compute allowed_classes per result in case the model was
                    # swapped at runtime and self.class_names is stale.
                    if result.names:
                        allowed_classes = self._effective_allowed(mode, set(result.names.values()))
                    for box in boxes:
                        cls_id_num = int(box.cls[0].item())
                        cls_name = result.names.get(cls_id_num, f"class_{cls_id_num}")
                        confidence = float(box.conf[0].item())

                        if allowed_classes is not None and cls_name not in allowed_classes:
                            continue

                        # Extract absolute bounding box (xywh)
                        xywh = box.xywh[0].tolist()  # center_x, center_y, width, height
                        cx, cy, w, h = xywh[0], xywh[1], xywh[2], xywh[3]

                        # Convert center xywh to top-left xywh and normalize to [0, 1]
                        top_left_x = max(0.0, min(1.0, (cx - w / 2.0) / img_w))
                        top_left_y = max(0.0, min(1.0, (cy - h / 2.0) / img_h))
                        norm_w = max(0.0, min(1.0, w / img_w))
                        norm_h = max(0.0, min(1.0, h / img_h))

                        detections.append({
                            "classId": cls_name,
                            "confidence": round(confidence, 3),
                            "box": {
                                "x": round(top_left_x, 4),
                                "y": round(top_left_y, 4),
                                "width": round(norm_w, 4),
                                "height": round(norm_h, 4),
                            }
                        })
                return detections
            except Exception as e:
                # If a real model is loaded but inference fails, return empty rather
                # than fake detections that would mislead the mobile app.
                logger.error(f"YOLO detection error: {e}")
                return []

        # Fallback detector -- only reached when model weights are unavailable
        return self._fallback_detect(mode, allowed_classes)

    def _fallback_detect(self, mode: str, allowed_classes: Optional[Set[str]]) -> List[Dict[str, Any]]:
        """
        Fallback simulation generator when offline or without weights.
        """
        logger.warning(
            f"[FALLBACK] '{self.model_name}' could not be loaded; producing random test detections. "
            f"load_error={self.load_error}"
        )
        pool = list(allowed_classes) if allowed_classes else ["chair", "person", "cell phone", "bottle", "laptop"]
        count = int(np.random.randint(1, 4))
        detections = []
        for _ in range(count):
            cls_name = str(np.random.choice(pool))
            w = float(round(np.random.uniform(0.18, 0.45), 4))
            h = float(round(np.random.uniform(0.18, 0.45), 4))
            x = float(round(np.random.uniform(0.05, 1.0 - w - 0.05), 4))
            y = float(round(np.random.uniform(0.05, 1.0 - h - 0.05), 4))
            conf = float(round(np.random.uniform(0.65, 0.95), 3))

            detections.append({
                "classId": cls_name,
                "confidence": conf,
                "box": {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                }
            })
        return detections
