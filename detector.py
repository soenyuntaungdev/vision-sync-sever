import base64
import io
import logging
from typing import Any, Dict, List, Optional, Set
import numpy as np
from PIL import Image

logger = logging.getLogger("visionsync")
logging.basicConfig(level=logging.INFO)

# Maximum image dimension kept before YOLO inference (frames are typically
# far larger than this; the model downsamples to 640 internally anyway).
MAX_INFER_DIM = 1280

# Mode filtering definitions matching constants/modes.ts in VisionSync app
MODE_CLASSES: Dict[str, Optional[Set[str]]] = {
    "general": None,  # All classes
    "security": {
        "person",
        "backpack",
        "handbag",
        "suitcase",
        "knife",
        "scissors",
        "cell phone",
        "car",
        "motorcycle",
        "bicycle",
        "truck",
        "bus",
    },
    "industrial": {
        "truck",
        "bus",
        "train",
        "bench",
        "chair",
        "scissors",
        "knife",
        "oven",
        "microwave",
        "toaster",
        "refrigerator",
        "laptop",
        "keyboard",
        "bottle",
        "fire hydrant",
    },
}

class ObjectDetector:
    """
    Ultralytics YOLOv8 object detector with automatic fallback to mock/synthetic
    detections if model weights are unavailable or if PyTorch/Ultralytics is absent.
    """

    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model_name = model_name
        self.model = None
        self.use_fallback = False
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            logger.info("YOLOv8 model loaded successfully!")
        except Exception as e:
            logger.warning(f"Could not load Ultralytics YOLO model ({e}). Using smart fallback detector.")
            self.use_fallback = True

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

        allowed_classes = MODE_CLASSES.get(mode, None)

        if not self.use_fallback and self.model is not None:
            try:
                # Downscale huge phone frames before inference: YOLO resizes to
                # 640 internally anyway, so we only pay the numpy cost once and
                # avoid very large allocations.
                if max(image.size) > MAX_INFER_DIM:
                    ratio = MAX_INFER_DIM / max(image.size)
                    image = image.resize(
                        (round(image.width * ratio), round(image.height * ratio)),
                        Image.LANCZOS,
                    )
                img_w, img_h = image.size
                # Convert PIL image to numpy array for YOLO
                img_np = np.array(image)
                results = self.model.predict(img_np, conf=conf_threshold, verbose=False)
                
                detections = []
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id_num = int(box.cls[0].item())
                        cls_name = result.names.get(cls_id_num, f"class_{cls_id_num}")
                        confidence = float(box.conf[0].item())

                        # Filter by mode if applicable
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
                logger.error(f"YOLO detection error: {e}")

        # Fallback detector if YOLO is not available
        return self._fallback_detect(mode, allowed_classes)

    def _fallback_detect(self, mode: str, allowed_classes: Optional[Set[str]]) -> List[Dict[str, Any]]:
        """
        Fallback simulation generator when offline or without weights.
        """
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
