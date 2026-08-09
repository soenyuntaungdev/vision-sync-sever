"""
VisionSync — Continuous Fine-Tuning Script (Automated, One-Click)

မူလ ၈၀ မျိုး ပါသည့် .pt ဖိုင်နှင့် ပစ္စည်းအသစ်တွေကို .pt တစ်ခုတည်းထဲ ပေါင်းစပ်လိုက်သော
အလိုအလျောက် script ဖြစ်ပါသည်။ လက်နဲ့ data.yaml မပြင်တော့ပါနဲ့။

အသုံးပြုပုံ (၃) မျိုးရှိပါသည် —

မျိုး (က) အလွယ်ကူဆုံး — Master Dataset auto တည်ဆောက်မှ စအဆုံးထိ
    0. အရင်ဆုံး PyYAML တပ်ဆင်ထားပါ
        pip install pyyaml ultralytics

    1. မိမိရှိပြီးသား .pt ဖိုင်ကို backend/models/ ထဲကူးထည့်ပါ၊ ဥပမာ
        backend/models/my_old_80_classes.pt

    2. ပစ္စည်းအသစ် dataset တစ်ခုကို Roboflow မှ export (YOLOv8 format) လုပ်ပြီး
        backend/dataset/myanmarpadauk/   ဒီလို folder ထဲထည့်ပါ
       (ထိုထဲမှာ data.yaml, train/images, train/labels, valid/... တွေပါရမည်)

    3. အောက်ပါ Command တစ်ကို ရိုက်ရုံဖြင့် အလုပ်အကုန် အလိုအလျောက် လုပ်ပေးမည် —

        python train_custom.py continue \
            --base models/my_old_80_classes.pt \
            --new-source dataset/myanmarpadauk \
            --class-name myanmar_padauk_coin \
            --epochs 20

    ရလဒ် —
        - Master dataset auto တည်ဆောက်ပြီး class အသစ်ကို ID 80+ ဖြင့် auto ထည့်
        - ပစ္စည်းအသစ်၏ labels ကို auto shift ပြုလုပ်ပြီး master ထဲ merge
        - Base .pt ကို မူတည်ပြီး fine-tune (epochs 20, lr0=0.001)
        - .pt အသစ် = 80 မူလမျိုး + အသစ် အားလုံးပါဝင်မည်
        - runs/detect/visionsync_master/weights/best.pt
        +  backup အဖြစ် models/visionsync_master_YYYYMMDD_HHMMSS_best.pt


မျိုး (ခ) Base မှတိုက်ရိုက် Train (Master Dataset ရှိပြီးသားဆိုရင်)
        python train_custom.py direct \
            --base models/my_old_80_classes.pt \
            --yaml dataset/master/data.yaml \
            --epochs 20


မျိုး (ဂ) သီးခြား Tools များ —
    python train_custom.py add-class --yaml dataset/master/data.yaml --name my_new_thing
    python train_custom.py merge   --master dataset/master --source dataset/x --class-name x
    python train_custom.py info    --pt models/my_old_80_classes.pt
"""

import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BACKEND_DIR)

from master_builder import (  # noqa: E402
    add_class_to_yaml,
    continuous_finetune,
    extract_model_info,
    merge_dataset_into_master,
    MASTER_DIR,
    _ensure_master_structure,
)


def cmd_continue(args: argparse.Namespace) -> int:
    """
    တစ်ခါတည်း: master တည်ဆောက် → class ထည့် → dataset merge → fine-tune
    """
    base_pt = args.base
    new_src = args.new_source
    class_name = args.class_name

    if not os.path.isfile(base_pt):
        print(f"❌ Base .pt မတွေ့ရှိပါ: {base_pt}")
        return 1
    if not os.path.isdir(new_src):
        print(f"❌ New source folder မတွေ့ရှိပါ: {new_src}")
        return 2

    info = extract_model_info(base_pt)
    if not info.get("ok"):
        print(f"❌ Base model info မရနိုင်ပါ: {info.get('message')}")
        return 3
    print(f"ℹ️  Base .pt မှာ class {info['nc']} မျိုးပါသည်။")

    _ensure_master_structure(MASTER_DIR)
    master_yaml = os.path.join(MASTER_DIR, "data.yaml")

    res_merge = merge_dataset_into_master(MASTER_DIR, new_src, class_name)
    if not res_merge["ok"]:
        print(f"❌ Dataset merge မအောင်မြင်ပါ: {res_merge['message']}")
        return 4
    print(f"✅ {res_merge['message']}")

    res_train = continuous_finetune(
        base_model_path=base_pt,
        data_yaml_path=master_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        run_name=args.name,
    )
    if not res_train["ok"]:
        print(f"❌ Training မအောင်မြင်ပါ: {res_train['message']}")
        return 5
    print(f"🎉 {res_train['message']}")
    print(f"   → Best  : {res_train['best_pt']}")
    if res_train.get("archived_pt"):
        print(f"   → Backup: {res_train['archived_pt']}")
    return 0


def cmd_direct(args: argparse.Namespace) -> int:
    if not os.path.isfile(args.base):
        print(f"❌ Base .pt မတွေ့ရှိပါ: {args.base}")
        return 1
    if not os.path.isfile(args.yaml):
        print(f"❌ data.yaml မတွေ့ရှိပါ: {args.yaml}")
        return 2
    res = continuous_finetune(
        base_model_path=args.base,
        data_yaml_path=args.yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        run_name=args.name,
    )
    print(json.dumps({k: v if k != "names" else f"[{len(v)} classes]" for k, v in res.items()},
                     ensure_ascii=False, indent=2))
    return 0 if res["ok"] else 5


def main() -> int:
    parser = argparse.ArgumentParser(description="VisionSync Continuous Fine-Tuning (Auto YAML)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("continue", help="မှ A-Z အထိ အလိုအလျောက် (Master + Merge + Train)")
    c.add_argument("--base", required=True, help="မူလ ၈၀ မျိုး ပါသည့် .pt ဖိုင်လမ်းကြောင်း")
    c.add_argument("--new-source", required=True, help="Dataset အသစ်ရရှိသော folder (data.yaml ပါသော)")
    c.add_argument("--class-name", required=True, help="ပစ္စည်းအသစ်၏ နာမည် (ဥပမာ myanmar_coin)")
    c.add_argument("--epochs", type=int, default=20)
    c.add_argument("--imgsz", type=int, default=640)
    c.add_argument("--batch", type=int, default=16)
    c.add_argument("--lr0", type=float, default=0.001)
    c.add_argument("--name", default="visionsync_master")

    d = sub.add_parser("direct", help="Master ရှိပြီးသားဖြစ်ပါက YAML ဖြင့် တိုက်ရိုက် Train")
    d.add_argument("--base", required=True)
    d.add_argument("--yaml", required=True)
    d.add_argument("--epochs", type=int, default=20)
    d.add_argument("--imgsz", type=int, default=640)
    d.add_argument("--batch", type=int, default=16)
    d.add_argument("--lr0", type=float, default=0.001)
    d.add_argument("--name", default="visionsync_master")

    a = sub.add_parser("add-class", help="data.yaml ထဲ Class အသစ်သာ ထည့်")
    a.add_argument("--yaml", required=True)
    a.add_argument("--name", required=True)

    m = sub.add_parser("merge", help="Dataset အသစ်ကို Master ထဲသာ Merge လုပ် (Train မလုပ်သေး)")
    m.add_argument("--master", required=True)
    m.add_argument("--source", required=True)
    m.add_argument("--class-name", required=True)
    m.add_argument("--source-ids", default=None)

    i = sub.add_parser("info", help=".pt ဖိုင်ထဲရှိ classes / nc ကြည့်ရန်")
    i.add_argument("--pt", required=True)

    args = parser.parse_args()
    if args.cmd == "continue":
        return cmd_continue(args)
    if args.cmd == "direct":
        return cmd_direct(args)
    if args.cmd == "add-class":
        res = add_class_to_yaml(args.yaml, args.name)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["ok"] else 1
    if args.cmd == "merge":
        ids = None
        if args.source_ids:
            try:
                ids = [int(x.strip()) for x in args.source_ids.split(",") if x.strip()]
            except Exception:
                print("❌ --source-ids format မှားနေပါသည်။ ဥပမာ 0,1")
                return 2
        res = merge_dataset_into_master(args.master, args.source, args.class_name, ids)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res["ok"] else 3
    if args.cmd == "info":
        res = extract_model_info(args.pt)
        if res.get("ok"):
            print(f"nc={res['nc']}   size={res['size_kb']} KB   path={res['path']}")
            for i, n in enumerate(res["names"]):  # type: ignore[arg-type]
                print(f"  {i:>3}: {n}")
        else:
            print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0 if res.get("ok") else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
