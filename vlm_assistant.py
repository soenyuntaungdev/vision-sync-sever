import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("visionsync.vlm")

def _load_env_file():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.isfile(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

_load_env_file()

# Gemini API Key loaded from environment or .env file
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

GEMINI_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-flash-latest"
]


class VLMAssistant:
    """
    Vision-Language Model (VLM) Assistant powered by Google Gemini Vision.
    Provides highly accurate natural language scene description,
    visual question answering (VQA), and zero-shot object detection in Myanmar language.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or DEFAULT_GEMINI_API_KEY
        self.is_configured = bool(self.api_key and not self.api_key.startswith("YOUR_"))
        logger.info(f"Gemini VLM Assistant initialized (configured={self.is_configured})")

    def _call_gemini(
        self,
        prompt: str,
        image_base64: Optional[str] = None,
        response_json: bool = False,
        timeout: int = 25
    ) -> Optional[Dict[str, Any]]:
        """
        Calls Google Gemini Vision API with automatic fallback across available Flash models.
        """
        if not self.is_configured:
            return None

        # Clean base64 string
        clean_b64 = image_base64 or ""
        mime_type = "image/jpeg"
        if "data:image/" in clean_b64 and ";base64," in clean_b64:
            header, clean_b64 = clean_b64.split(";base64,", 1)
            mime_type = header.split("data:image/")[1]
        elif "," in clean_b64:
            clean_b64 = clean_b64.split(",", 1)[1]

        clean_b64 = clean_b64.strip().replace("\n", "").replace("\r", "")

        parts: List[Dict[str, Any]] = [{"text": prompt}]
        if clean_b64:
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": clean_b64
                }
            })

        payload: Dict[str, Any] = {
            "contents": [{"parts": parts}]
        }
        if response_json:
            payload["generationConfig"] = {"response_mime_type": "application/json"}

        for model_name in GEMINI_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"
            try:
                res = requests.post(url, json=payload, timeout=timeout)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        text = candidates[0]["content"]["parts"][0].get("text", "")
                        return {"ok": True, "text": text, "model": model_name}
                else:
                    logger.warning(f"Gemini model {model_name} returned status {res.status_code}: {res.text[:120]}")
            except Exception as e:
                logger.warning(f"Gemini API request failed for {model_name}: {e}")

        return None

    def describe_scene(self, image_base64: str) -> Dict[str, Any]:
        """
        Generates a natural, clear Myanmar scene description for visually impaired users.
        """
        prompt = (
            "You are an assistive AI for visually impaired users (မျက်မမြင်/အမြင်အာရုံချို့တဲ့သူများအတွက် အထောက်အကူပြု AI). "
            "Look at this camera image and describe the scene clearly in natural Myanmar (Burmese) language. "
            "Mention: "
            "1. What is directly in front of the user (e.g. table, chair, person, door, stairs, obstacles, distance) "
            "2. Any important items, text, signs, or currency notes visible "
            "3. Any hazards or obstacles to watch out for. "
            "Keep the response concise (2-4 sentences), encouraging, and helpful for navigation."
        )

        res = self._call_gemini(prompt, image_base64=image_base64)
        if res and res.get("ok"):
            return {
                "ok": True,
                "description": res["text"].strip(),
                "model": res.get("model", "gemini-flash"),
                "source": "google_gemini_vision"
            }

        return {
            "ok": True,
            "description": "ရုပ်ပုံကို လက်ခံရရှိပြီး ခွဲခြမ်းစိတ်ဖြာနေပါသည်။",
            "source": "fallback"
        }

    def ask_question(self, image_base64: str, question: str) -> Dict[str, Any]:
        """
        Answers a user's natural language question about the image in Myanmar language.
        """
        prompt = (
            f"User Question: '{question}'\n\n"
            "Answer the user's question accurately based on what you see in the provided image. "
            "Always reply in polite, fluent Myanmar (Burmese) language."
        )

        res = self._call_gemini(prompt, image_base64=image_base64)
        if res and res.get("ok"):
            return {
                "ok": True,
                "question": question,
                "answer": res["text"].strip(),
                "model": res.get("model", "gemini-flash"),
                "source": "google_gemini_vision"
            }

        return {
            "ok": False,
            "question": question,
            "error": "Gemini API နှင့် ချိတ်ဆက်ရာတွင် အဆင်မပြေဖြစ်နေပါသည်။"
        }

    def detect_objects(self, image_base64: str, conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detects objects in the image using Gemini Vision and returns normalized bounding boxes.
        Format matches VisionSync's ObjectDetector format:
        [
          {
            "classId": "person",
            "confidence": 0.95,
            "box": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.5}
          }
        ]
        """
        prompt = (
            "Detect all salient objects, obstacles, people, and items in this image. "
            "Return ONLY a JSON array with this exact structure: "
            "[\n"
            "  {\n"
            "    \"classId\": \"name of object in english lowercase (e.g. person, chair, bottle, cell phone, stairs, car)\",\n"
            "    \"confidence\": 0.95,\n"
            "    \"box\": {\"x\": 0.15, \"y\": 0.20, \"width\": 0.40, \"height\": 0.50}\n"
            "  }\n"
            "]\n"
            "Note: 'x' and 'y' are normalized top-left coordinates [0.0 to 1.0]. "
            "'width' and 'height' are normalized dimensions [0.0 to 1.0]."
        )

        res = self._call_gemini(prompt, image_base64=image_base64, response_json=True)
        if res and res.get("ok"):
            try:
                raw_json = res["text"].strip()
                # Clean code fences if present
                if raw_json.startswith("```"):
                    raw_json = re.sub(r"^```[a-zA-Z]*\n", "", raw_json)
                    raw_json = re.sub(r"```$", "", raw_json).strip()

                parsed = json.loads(raw_json)
                if isinstance(parsed, dict) and "detections" in parsed:
                    parsed = parsed["detections"]

                if isinstance(parsed, list):
                    clean_detections: List[Dict[str, Any]] = []
                    for item in parsed:
                        if not isinstance(item, dict):
                            continue
                        cls_name = str(item.get("classId") or item.get("label") or "object")
                        conf = float(item.get("confidence", 0.9))
                        if conf < conf_threshold:
                            continue
                        box = item.get("box", {})
                        if isinstance(box, dict):
                            x = float(max(0.0, min(1.0, box.get("x", 0.0))))
                            y = float(max(0.0, min(1.0, box.get("y", 0.0))))
                            w = float(max(0.0, min(1.0, box.get("width", 0.1))))
                            h = float(max(0.0, min(1.0, box.get("height", 0.1))))
                            clean_detections.append({
                                "classId": cls_name,
                                "confidence": round(conf, 3),
                                "box": {
                                    "x": round(x, 4),
                                    "y": round(y, 4),
                                    "width": round(w, 4),
                                    "height": round(h, 4)
                                }
                            })
                    return clean_detections
            except Exception as e:
                logger.error(f"Failed to parse Gemini detection JSON: {e}")

        return []
