# Training Dataset Reading Issue

> **Workspace:** e:\VisionSync

---

**User**

ဘာလို့ training ထားတဲ့ dataset မှာ ရင် ဟာတွေပါနေတာကို ဘာလို့မဖတ်တာလဲ

**Assistant**

ကျွန်ုပ်တို့ ပထမဦးစွာ backend code ထဲမှာ dataset loading နှင့် training logic တွေကို စစ်ဆေးကြည့်ပါမယ်။ ပထမဆုံး main.py ဖိုင်နှင့် dataset/training 관련 code များကို ရှာဖွေကြည့်မည်။


အသေးစိတ်ပိုမိုလေ့လာရန် dataset_utils.py, training_manager.py, master_builder.py ဖိုင်တွေကို ဖတ်ကြည့်ပါမယ်။


`_shift_label_file` function ကို ပထမဦးစွာ စစ်ဆေးကြည့်ရန် လိုပါသည်။ သင့်ရဲ့ dataset ထဲမှာ person/ dog classes နှစ်မျိုးလုံးပါနေပြီး ဘယ်လို mapping လုပ်နေလဲ ဆိုတာ သိရန် လိုပါတယ်။


အခု ပြဿနာ ပိုမိုရှင်းလင်းလာပါပြီ။ Dataset ထဲမှာ person/dog classes တွေပါနေပါတယ်ဆိုရင် ဘယ်န

---

