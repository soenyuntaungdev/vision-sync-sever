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

# COCO ၈၀ မျိုး၏ နာမည်များ။ MODE_CLASSES filter များကို ဤနာမည်များအပေါ်မှာ
# အခြေခံရေးထားသဖြင့်၊ model တစ်ခုမှာ ဤစာရင်းထဲ မပါသော class (= custom class)
# တွေ့ရင် mode filter က ဖြတ်မပစ်စေရန် သုံးသည်။
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

    def __init__(self, model_name: str = "yolov8n.pt", allow_fallback: bool = True):
        self.model_name = model_name
        self.model = None
        self.use_fallback = False
        self.load_error: Optional[str] = None
        self.class_names: List[str] = []
        # allow_fallback=False ဆိုရင် model load မရလျှင် Exception ပစ်မည်။
        # (Activate လုပ်ချိန်မှာ "အောင်မြင်တယ်" လို့ပြပြီး တကယ်တမ်း fake detection
        #  ထုတ်နေတာမျိုး မဖြစ်စေရန် ဖြစ်သည်။)
        self.allow_fallback = allow_fallback
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLOv8 model: {self.model_name}...")
            self.model = YOLO(self.model_name)
            names_attr = getattr(getattr(self.model, "model", None), "names", None)
            if names_attr is None:
                names_attr = getattr(self.model, "names", None)
            if isinstance(names_attr, dict):
                self.class_names = [str(names_attr[k]) for k in sorted(names_attr.keys())]
            elif isinstance(names_attr, (list, tuple)):
                self.class_names = [str(n) for n in names_attr]
            logger.info(
                f"YOLOv8 model loaded successfully! nc={len(self.class_names)} "
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
        Mode filter ကို လက်ရှိ model ရဲ့ class များနှင့် ကိုက်ညီအောင် တွက်ပေးသည်။

        - mode == "general" → filter မရှိ (None)
        - Custom training လုပ်ထားသော class (COCO ၈၀ ထဲ မပါသော နာမည်) မှန်သမျှကို
          အမြဲ ခွင့်ပြုသည်။ ဒါမှသာ fine-tune လုပ်ပြီး ထည့်လိုက်တဲ့ class အသစ်ဟာ
          security / industrial mode မှာ ပျောက်မသွားမည်။
        - Model ရဲ့ class တွေထဲ mode စာရင်းနဲ့ လုံးဝ မဆိုင်ရင် filter ကို ပိတ်ပေးသည်။
        """
        base = MODE_CLASSES.get(mode, None)
        if base is None:
            return None
        if not model_names:
            return base
        custom_names = model_names - COCO_CLASS_NAMES
        allowed = set(base) | custom_names
        if not (model_names & allowed):
            # ဒီ model နဲ့ ဒီ mode လုံးဝမကိုက်ဘူး → အားလုံးပြပေးမယ် (ဘာမှမပေါ်တာထက် ကောင်းသည်)
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
                    # Model ကို runtime မှာ swap လုပ်လိုက်တဲ့အခါ self.class_names က
                    # ရှေ့ကတန်ဖိုးဖြစ်နေနိုင်လို့ result.names ကို အခြေခံပြီး ထပ်တွက်သည်။
                    if result.names:
                        allowed_classes = self._effective_allowed(mode, set(result.names.values()))
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
                # Model တစ်ခု တကယ် load ဖြစ်နေပြီးမှ inference မှားတာဆိုရင်
                # fake detection မထုတ်ဘဲ ဗလာပြန်ပေးမည်။ (fake data က mobile app ဘက်မှာ
                # "model အလုပ်လုပ်နေတယ်" ဟု ထင်မှားစေသည်)
                logger.error(f"YOLO detection error: {e}")
                return []

        # Fallback detector — model weights လုံးဝ မရှိမှသာ
        return self._fallback_detect(mode, allowed_classes)

    def _fallback_detect(self, mode: str, allowed_classes: Optional[Set[str]]) -> List[Dict[str, Any]]:
        """
        Fallback simulation generator when offline or without weights.
        """
        logger.warning(
            f"[FALLBACK] '{self.model_name}' ကို load မရသဖြင့် စမ်းသပ် (random) detection "
            f"ထုတ်နေပါသည် — load_error={self.load_error}"
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
