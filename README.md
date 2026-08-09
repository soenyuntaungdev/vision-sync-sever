# VisionSync AI Backend Server

VisionSync မိုဘိုင်း အက်ပလီကေးရှင်းအတွက် **FastAPI + Ultralytics YOLOv8** Real-time Object Detection Backend Server ဖြစ်ပါသည်။

---

## 📌 Features

- **Real-time YOLOv8 Object Detection**: ကင်မရာမှ base64 image frames များကို လက်ခံပြီး အရာဝတ္ထုများကို 0.35+ confidence ဖြင့် ရှာဖွေပေးသည်။
- **Mode-based Filtering**:
  - `general`: COCO 80 အရာဝတ္ထု အမျိုးအစား အားလုံး။
  - `security`: လူ၊ အိတ်၊ ဓား၊ ကတ်ကြေး၊ ဖုန်း၊ ယာဉ်များ။
  - `industrial`: စက်ကိရိယာ၊ အီလက်ထရောနစ်၊ အိမ်သုံးပစ္စည်းများ။
- **Normalized Bounding Box Output**: `(x, y, width, height)` 0.0 - 1.0 တန်ဖိုးများဖြင့် မိုဘိုင်း screen တွင် တိုက်ရိုက် ရေးဆွဲနိုင်ရေး တွက်ချက်ပေးသည်။
- **Offline / Fallback Support**: YOLOv8 weights ဒေါင်းလုဒ် မပြီးသေးပါက သို့မဟုတ် offline ဖြစ်နေပါက Server မရပ်သွားဘဲ စမ်းသပ် detections များ ထုတ်ပေးမည်။
- **User Feedback / Reports API**: မှားယွင်းသော detections အစီရင်ခံစာများကို Dataset retraining ပြုလုပ်ရန် `reports_log.json` တွင် သိမ်းဆည်းပေးသည်။

---

## 🚀 Quick Start Guide

### 1. Requirements သွင်းယူခြင်း

```bash
cd backend
pip install -r requirements.txt
```

### 2. Backend Server စတင် run ခြင်း

```bash
python main.py
```

သို့မဟုတ်:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔗 API Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server Health Status & YOLO Model Check |
| `POST` | `/detect` | Base64 Image Frame ရယူပြီး Detection ရလဒ်များ ပြန်ပေးခြင်း |
| `POST` | `/reports` | Detection မှားယွင်းမှု အစီရင်ခံစာ သိမ်းဆည်းခြင်း |
| `GET` | `/reports` | သိမ်းဆည်းထားသော အစီရင်ခံစာများ ကြည့်ရှုခြင်း |
| `GET` | `/docs` | Interactive Swagger API Documentation |

---

## 📱 Mobile App ထဲတွင် ချိတ်ဆက်နည်း

1. **VisionSync App** ကို ဖွင့်ပါ။
2. **ဆက်တင် (Settings)** Tab သို့ သွားပါ။
3. **Backend ချိတ်ဆက်မှု (Backend Connection)** ကို နှိပ်ပါ။
4. Server URL တွင် မိမိ ကွန်ပျူတာ၏ IP Address ကို ထည့်ပါ (ဥပမာ- `http://192.168.1.10:8000`)။
5. **ချိတ်ဆက်မှု စမ်းသပ်ရန်** ကို နှိပ်ပြီး **Real Backend သုံးမည်** ကို On ပေးပါ။
