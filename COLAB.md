# VisionSync — Google Colab မှာ Run ရန်

## ⚠️ အရင် Colab script မှာ ဘာမှားနေခဲ့လဲ

သင့်အရင် script က cell run တိုင်း ဒီလိုလုပ်ခဲ့သည် —

```python
!rm -rf visionsync_repo
!git clone $GITHUB_REPO_URL visionsync_repo
```

Colab ရဲ့ `/content` က **ephemeral** (session ပြီးရင် ပျက်သွားသည်) ဖြစ်ပြီး၊
ဒီ `rm -rf` က ကျန်တာကိုပါ ရှင်းပစ်လိုက်သည်။ ဒါကြောင့် —

- Training ပြီးလို့ ထွက်လာတဲ့ `runs/detect/.../best.pt` → **ပျောက်**
- `models/` ထဲ archive လုပ်ထားတဲ့ `.pt` → **ပျောက်**
- Upload လုပ်ထားတဲ့ `dataset/` → **ပျောက်**
- Activate လုပ်ထားတဲ့ မှတ်တမ်း → **ပျောက်**

ဒါကြောင့် cell ပြန် run တိုင်း မူလ `yolov8n.pt` ကို ပြန်ရောက်နေခြင်း ဖြစ်သည်။

**ဖြေရှင်းနည်း:** Google Drive ကို mount လုပ်ပြီး `dataset/`, `runs/`, `models/`,
`active_model.json` တို့ကို Drive ပေါ်မှာ ထားလိုက်ရန်။ အောက်က script က အဲဒါကို
အလိုအလျောက် လုပ်ပေးသည်။

---

## ✅ ပြင်ပြီးသား Colab Cell (ဒါကို copy ကူးပါ)

```python
# ============================================================
# VISION SYNC BACKEND — GOOGLE COLAB + NGROK
#   dataset/ runs/  → local disk  (မြန် + Drive ပြုတ်လည်း training မပျက်)
#   models/ uploads/→ Google Drive (ရလဒ် .pt မပျောက်စေရန်)
# ============================================================

NGROK_TOKEN     = "သင့် ngrok token"
GITHUB_REPO_URL = "https://github.com/soenyuntaungdev/vision-sync-sever.git"
USE_DRIVE       = True   # False ဆိုရင် session ပြီးတာနဲ့ .pt တွေ ပျောက်မည်

# ---------- Step 1: Packages ----------
!pip install -q fastapi "uvicorn[standard]" ultralytics pydantic python-multipart \
                pillow "numpy<2.0" opencv-python-headless pyyaml pyngrok nest_asyncio 2>&1 | tail -3

import os, re, shutil, subprocess, time

REPO       = '/content/visionsync_repo'
LOCAL_DS   = '/content/local_dataset'
LOCAL_RUNS = '/content/local_runs'

# ---------- Step 2: Google Drive mount ----------
DRIVE = None
if USE_DRIVE:
    from google.colab import drive
    if not os.path.isdir('/content/drive/MyDrive'):
        drive.mount('/content/drive')
    if not os.path.isdir('/content/drive/MyDrive'):
        raise SystemExit('❌ Drive mount မအောင်မြင်ပါ')
    DRIVE = '/content/drive/MyDrive/visionsync_data'
    os.makedirs(DRIVE, exist_ok=True)
    print(f'[OK] Drive: {DRIVE}')

# ---------- Step 3: Repo (code ကိုသာ အသစ်ယူမည်) ----------
os.chdir('/content')
if os.path.isdir(os.path.join(REPO, '.git')):
    print('[..] repo ရှိပြီးသား — git pull လုပ်နေသည်')
    subprocess.run(['git', '-C', REPO, 'fetch', '--all'], check=False)
    subprocess.run(['git', '-C', REPO, 'reset', '--hard', 'origin/main'], check=False)
else:
    shutil.rmtree(REPO, ignore_errors=True)
    subprocess.run(['git', 'clone', GITHUB_REPO_URL, REPO], check=True)

BACKEND = REPO
if os.path.isfile(os.path.join(REPO, 'backend', 'main.py')):
    BACKEND = os.path.join(REPO, 'backend')
os.chdir(BACKEND)
print(f'[OK] Working dir: {os.getcwd()}')

# ---------- Step 4: Folder layout ----------
def relink(link_path, target):
    """link_path → target ချိတ်မည်။ ဖိုင်အစစ်ရှိရင် ဖျက်မပစ်ဘဲ target ထဲ ရွှေ့ပြီးမှ ချိတ်သည်။"""
    os.makedirs(target, exist_ok=True)
    name = os.path.basename(link_path)
    if os.path.islink(link_path):
        if os.path.realpath(link_path) == os.path.realpath(target):
            print(f'  = {name}/ (ချိတ်ပြီးသား)');  return
        os.unlink(link_path)
    elif os.path.isdir(link_path):
        moved = 0
        for item in os.listdir(link_path):
            src, dst = os.path.join(link_path, item), os.path.join(target, item)
            if not os.path.exists(dst):
                shutil.move(src, dst); moved += 1
        shutil.rmtree(link_path, ignore_errors=True)
        if moved: print(f'  ! {name}/ ထဲက {moved} ခုကို ကယ်ပြီး ရွှေ့ပြီး')
    os.symlink(target, link_path)
    print(f'  → {name}/ -> {target}')

# 4a. dataset ကို Drive ကနေ local disk သို့ restore
#     (Drive ကနေ တိုက်ရိုက် train ရင် mount ပြုတ်ချိန် "Image Not Found" တက်သည်)
if DRIVE:
    drive_ds = f'{DRIVE}/dataset'
    os.makedirs(drive_ds, exist_ok=True)
    if not os.path.isdir(LOCAL_DS):
        print('[..] dataset ကို Drive → local ကူးနေသည် (၁-၂ မိနစ်)')
        shutil.copytree(drive_ds, LOCAL_DS, dirs_exist_ok=True)
    print(f'[OK] local dataset: {LOCAL_DS}')

# 4b. symlink များ ချိတ်
relink(os.path.join(BACKEND, 'dataset'), LOCAL_DS)
relink(os.path.join(BACKEND, 'runs'),    LOCAL_RUNS)
if DRIVE:
    relink(os.path.join(BACKEND, 'models'),  f'{DRIVE}/models')
    relink(os.path.join(BACKEND, 'uploads'), f'{DRIVE}/uploads')

    am_target, am_link = f'{DRIVE}/active_model.json', os.path.join(BACKEND, 'active_model.json')
    if not os.path.exists(am_target):
        open(am_target, 'w').write('{"model_path": "yolov8n.pt"}')
    if os.path.lexists(am_link): os.remove(am_link)
    os.symlink(am_target, am_link)
    print(f'  → active_model.json -> {am_target}')

# 4c. data.yaml ရဲ့ path များကို relative ပြန်လုပ် (server က local အဖြစ် auto ပြန်ချိတ်မည်)
y = os.path.join(BACKEND, 'dataset', 'master', 'data.yaml')
if os.path.isfile(y):
    _s = open(y, encoding='utf-8').read()
    _r = {'path':'path: .', 'train':'train: images/train',
          'val':'val: images/val', 'test':'test: images/test'}
    open(y, 'w', encoding='utf-8').write(
        re.sub(r'^(path|train|val|test):.*$', lambda m: _r[m.group(1)], _s, flags=re.M))
    for _sp in ('train', 'val'):
        _c = os.path.join(BACKEND, 'dataset', 'master', 'labels', _sp + '.cache')
        if os.path.exists(_c): os.remove(_c)          # stale cache ဖျက်
        _d = os.path.join(BACKEND, 'dataset', 'master', 'images', _sp)
        if os.path.isdir(_d): print(f'  master {_sp}: {len(os.listdir(_d))} ပုံ')

# ---------- Step 5: အရင် server ကို ရပ် (port 8000 လွတ်စေရန်) ----------
subprocess.run(['pkill', '-9', '-f', 'uvicorn main:app'], check=False)
time.sleep(2)
import socket
_s = socket.socket(); _busy = _s.connect_ex(('127.0.0.1', 8000)) == 0; _s.close()
print('[!!] port 8000 မလွတ်သေးပါ — Runtime → Restart session' if _busy else '[OK] port 8000 လွတ်ပါပြီ')

# ---------- Step 6: ngrok ----------
from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_TOKEN
subprocess.run(['pkill', '-f', 'ngrok'], check=False)
ngrok.kill()

import nest_asyncio
nest_asyncio.apply()

tunnel = ngrok.connect(8000, bind_tls=True)
print('\n' + '=' * 70)
print(f'✅ PUBLIC URL (Mobile ထဲထည့်ရန်): {tunnel.public_url}')
print(f'    Health Check: {tunnel.public_url}/health')
print(f'    Training UI:  {tunnel.public_url}/training')
print('=' * 70 + '\n')

# ---------- Step 7: Server ----------
!uvicorn main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120
```

### 💾 Dataset ကို Drive ကို ပြန်သိမ်းရန် (zip အသစ်တင်ပြီးတိုင်း)

`dataset/` က local disk မှာ ရှိနေလို့ **session ပြီးရင် ပျောက်**ပါမယ်။
Zip အသစ်တင်ပြီး / COCO replay ထည့်ပြီးတိုင်း ဒီ cell ကို တစ်ခါ run ပါ —

```python
!rsync -a --delete /content/local_dataset/ /content/drive/MyDrive/visionsync_data/dataset/
print('✅ dataset ကို Drive ကို သိမ်းပြီး')
```

> ရလဒ် `.pt` ဖိုင်တွေကတော့ `models/` (Drive) ထဲ **အလိုအလျောက်** ရောက်နေပြီးသားမို့
> ဒီ cell မလုပ်လည်း မပျောက်ပါ။ Dataset ကိုပဲ သိမ်းတာပါ။
### ⚠️ `[Errno 98] address already in use` တက်ရင်

အရင် server က မသေဘဲ ကျန်နေတာပါ။ **အန္တရာယ်ရှိတာက** — ngrok URL က အလုပ်လုပ်နေဦးမှာမို့
"အဆင်ပြေတယ်" ထင်ရပေမယ့် တကယ်က **အဟောင်း code ကိုပဲ သုံးနေတာ** ဖြစ်ပါတယ်။
`git pull` နဲ့ ဆွဲထားတဲ့ ပြင်ဆင်မှုတွေ အသက်မဝင်ပါ။

Cell အသစ်တစ်ခုမှာ ဒါကို run ပါ —

```python
!pkill -9 -f "uvicorn main:app"
```

ပြီးမှ main cell ကို ပြန် run ပါ။ မရသေးရင် **Runtime → Restart session** လုပ်ပါ။

---

## 🔍 Training ပြီးရင် စစ်ရမည့် အဆင့်များ

### ၁။ Model တကယ် load ဖြစ်ရဲ့လား

```
GET {public_url}/health
```

```json
{
  "model_loaded": true,          // ← false ဆိုရင် detection တွေက အတုပါ
  "model_name": "models/visionsync_master_20260812_101500_best.pt",
  "load_error": null,            // ← error ရှိရင် ဒီမှာ ပေါ်မည်
  "nc": 81,
  "class_names": ["person", ..., "my_new_class"]
}
```

`model_loaded: false` ဆိုရင် `load_error` ကို ကြည့်ပါ။ Activate က အခု
မအောင်မြင်ရင် အမှားပြပါပြီ — အရင်လို "အောင်မြင်တယ်" ပြပြီး နောက်ကွယ်မှာ
random fake detection ထုတ်နေတာ မဟုတ်တော့ပါ။

### ၂။ Class အသစ် တကယ် ပါရဲ့လား

`/health` ရဲ့ `class_names` ထဲမှာ သင့် class အသစ် ပါရမည်။ မပါရင် dataset
merge မဖြစ်ခဲ့တာ (သို့) train လုပ်တဲ့ `data.yaml` မှားနေတာ ဖြစ်သည်။

### ၃။ Dataset မှာ class တစ်ခုချင်း label ရှိရဲ့လား

```
GET {public_url}/master/audit
```

`empty_classes` ထဲ ပါနေတဲ့ class တွေက fine-tune ပြီးရင် **ပျောက်သွားမည့်**
class တွေ ဖြစ်သည်။

### ၄။ Confidence နိမ့်နေလား

Dataset သေးရင် model က confidence နိမ့်တတ်သည်။ `/detect` မှာ `conf`
ထည့်ပြီး စမ်းကြည့်ပါ —

```json
{ "image": "...", "mode": "general", "conf": 0.15 }
```

---

## 📦 Dataset ကို ZIP နဲ့ Upload လုပ်ခြင်း

Zip ပုံစံ ၃ မျိုးလုံး အလိုအလျောက် ဖတ်ပေးပါသည် (test လုပ်ပြီးပါပြီ) —

| Zip ထဲက ပုံစံ | ဥပမာ |
|---|---|
| Roboflow (folder တစ်ခုနဲ့ ထုပ်ထား) | `MyCoins-1/data.yaml`, `MyCoins-1/train/images/…` |
| Roboflow (root မှာ တိုက်ရိုက်) | `data.yaml`, `train/images/…`, `valid/labels/…` |
| YOLOv8 standard | `data.yaml`, `images/train/…`, `labels/train/…` |

`valid/` → `val/` ပြောင်းတာ၊ nested folder ဖြေတာ၊ `data.yaml` ရဲ့ path တွေကို
ဒီစက်နဲ့ကိုက်အောင် ပြင်တာတွေ အကုန် အလိုအလျောက် လုပ်ပေးပါသည်။

### ⚠️ Class များစွာပါတဲ့ zip — `merge_mode`

Zip ထဲမှာ class တစ်ခုထက်ပိုပါရင် (ဥပမာ `['coin','note','card']`) ဘယ်လိုပေါင်းမလဲ
ရွေးလို့ရပါသည် —

| `merge_mode` | ရလဒ် |
|---|---|
| `auto` (default) | class ၁ ခုပဲဆိုရင် `collapse`၊ များနေရင် `per_class` |
| `per_class` | `coin`, `note`, `card` သုံးခုလုံးကို master ထဲ သီးသန့်ထည့် |
| `collapse` | သုံးခုလုံးကို သင်ပေးထားတဲ့ class နာမည် တစ်ခုတည်းအဖြစ် ပေါင်း |

```
POST /master/upload-and-merge   (multipart form)
  file=<your.zip>
  class_name=money
  merge_mode=per_class
  source_class_ids=0,2      ← (optional) ဒီ ID တွေပဲယူ၊ ကျန်တာ ဖြုတ်
```

> **အရင်က ဒီနေရာမှာ bug ရှိခဲ့သည်** — class ၃ မျိုးပါတဲ့ zip ကို merge လုပ်ရင်
> label ထဲမှာ master ရဲ့ `nc` ကို ကျော်တဲ့ class ID တွေ ဝင်သွားပြီး dataset
> ပျက်ခဲ့သည် (Ultralytics က "corrupt label" ပြပြီး ကျော်သွား ဒါမှမဟုတ် အမှိုက်
> model ထွက်လာသည်)။ အခု ပြင်ပြီးပါပြီ။

Merge ပြီးတိုင်း `GET /master/audit` နဲ့ စစ်ပါ — class တစ်ခုချင်း label
ဘယ်နှစ်ခုရှိလဲ ပြပေးပါသည်။

---

## 🧠 Class အသစ်ထည့်ရင် အဟောင်း ၈၀ မျိုး မပျောက်စေရန်

Ultralytics မှာ base model ရဲ့ `nc` နဲ့ dataset ရဲ့ `nc` မတူတာနဲ့ detection
head ကို **အသစ်ပြန်တည်ဆောက်** သည်။ ဒါကြောင့် class အသစ်တစ်ခုပဲပါတဲ့ dataset
နဲ့ train ရင် မူလ COCO ၈၀ မျိုးလုံး ပျောက်သွားသည် (catastrophic forgetting)။

လက်တွေ့မှာ ဒီနှစ်ခုကို လုပ်ပါ —

### ၁။ COCO Replay ထည့်ပါ ← **အရေးအကြီးဆုံး**

မူလ COCO class တွေရဲ့ ပုံအချို့ကို master dataset ထဲ ပြန်ထည့်ပေးလိုက်ရင်
fine-tune လုပ်တဲ့အခါ model က **class အသစ်ရော အဟောင်း ၈၀ ရော တစ်ပြိုင်တည်း**
သင်ယူသွားပါမည်။ တစ်ခါထည့်ရုံပါပဲ။

**Training UI မှာ** — `(၂-က) COCO Replay` ကတ်ထဲက **📥 COCO Replay ထည့်မည်**
ကို နှိပ်ပါ။

**API နဲ့ဆိုရင် —**

```
POST /master/add-coco-replay
{ "source": "val2017", "per_class": 30 }

GET  /master/replay-status      ← တိုးတက်မှု စစ်ရန်
DELETE /master/coco-replay      ← ဖျက်ရန်
```

**Colab terminal / CLI နဲ့ဆိုရင် —**

```bash
python coco_replay.py --source val2017 --per-class 30
```

| `source` | အရွယ် | Class ကာဗာ | ဘယ်အချိန်သုံး |
|---|---|---|---|
| `val2017` | ~၈၀၀ MB | ၈၀ လုံး ✅ | တကယ်သုံးမယ်ဆိုရင် ဒါကိုသုံးပါ |
| `coco128` | ~၇ MB | ၇၁ / ၈၀ | အမြန် စမ်းချင်ရင်သာ |

> Download က ပထမတစ်ခါသာ ဖြစ်ပါသည် — `.cache/coco/` ထဲ သိမ်းထားပါမည်။
> အထက်က Colab script က `.cache` ကို Drive မှာ ထားပေးလို့ session အသစ်မှာ
> ထပ်မ download ရပါ။

ရလဒ် (coco128 နဲ့ စမ်းထားတာ) —

```
merge မလုပ်ခင်:  Class 82 ခုအနက် 80 ခုမှာ label လုံးဝမရှိပါ  ❌
replay ထည့်ပြီး: Class 82 ခုအနက်  9 ခုမှာ label လုံးဝမရှိပါ  ✅
                 person=250  car=46  chair=35  myanmar_coin=8
```

`per_class` က "class တစ်ခုလျှင် ရည်မှန်း instance အရေအတွက်" ဖြစ်သည်။ ပုံတစ်ပုံမှာ
class များစွာ ပါတတ်လို့ `30` ဆိုရင် ပုံ ၅၀၀–၁၅၀၀ လောက် ရွေးပါမည်။ သင့် class
အသစ်ရဲ့ ပုံအရေအတွက်နဲ့ မျှမျှတတ ဖြစ်အောင် ချိန်ပါ — အသစ်က ၂၀၀ ပုံဆိုရင်
`per_class` ကို ၃၀–၅၀ လောက် ထားပါ။

### ၂။ `freeze` ကို သုံးပါ

Default `10` ဖြစ်သွားပါပြီ — backbone ကို ခဲထားပြီး head ကိုသာ သင်ပေးသည်။
Feature တွေ မပျက်ဘဲ ကျန်သည်။

```json
POST /master/start-direct-finetune
{ "base_model": "yolov8n.pt", "epochs": 30, "lr0": 0.001, "freeze": 10 }
```

`freeze: 0` ထားရင် အကုန် train မည် (dataset ကြီးမှသာ သင့်တော်သည်)။

---

## 📋 အစအဆုံး လုပ်ရမည့် အစီအစဉ်

```
၁။ COCO Replay ထည့်      →  POST /master/add-coco-replay {"source":"val2017"}
၂။ Class အသစ် zip upload  →  POST /master/upload-and-merge
၃။ စစ်ဆေး                →  GET  /master/audit        (empty_classes နည်းရမည်)
၄။ Fine-tune             →  POST /master/start-direct-finetune {"freeze":10}
၅။ Activate              →  POST /training/activate-model
၆။ အတည်ပြု               →  GET  /health   (model_loaded=true, class_names စစ်)
```
