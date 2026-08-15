(function () {
  "use strict";

  const API_BASE = "";
  const $ = (id) => document.getElementById(id);

  const modelsListEl = $("modelsList");
  const logBox = $("logBox");
  const tailLogsCb = $("tailLogs");

  // Master-specific elements
  const mZipFileInput = $("mZipFile");
  const mZipFileName = $("mZipFileName");
  const mFilePicker = document.querySelector(".tab-panel[data-panel=master] .file-picker");
  const mUploadForm = $("mUploadForm");
  const mUploadBtn = $("mUploadBtn");
  const mUploadMsg = $("mUploadMsg");
  const mDatasetName = $("mDatasetName");
  const mClassName = $("mClassName");
  const mRefreshMasterBtn = $("mRefreshMasterBtn");
  const mMasterInfo = $("mMasterInfo");
  const mMasterBadge = $("mMasterBadge");
  const mBaseModel = $("mBaseModel");
  const mStartBtn = $("mStartBtn");
  const mProgressFill = $("mProgressFill");
  const mStatusLine = $("mStatusLine");
  const mDurationText = $("mDurationText");
  const mResultRow = $("mResultRow");
  const mResultPt = $("mResultPt");
  const mActivateBtn = $("mActivateBtn");
  const mDownloadBtn = $("mDownloadBtn");

  let mSince = 0;
  let mStartedAt = null;
  let mPollTimer = null;
  let mDurationTimer = null;
  let mLastStartedAt = null;

  const M_STATUS_LABELS = {
    idle: "အနားယူနေသည်",
    running: "Fine-tune လုပ်နေသည်…",
    ok: "အောင်မြင်ပြီးပြီ",
    error: "အမှားဖြစ်နေသည်",
  };

  function setMsg(el, text, level) {
    el.textContent = text || "";
    el.classList.remove("is-ok", "is-err");
    if (level === "ok") el.classList.add("is-ok");
    if (level === "err") el.classList.add("is-err");
  }

  async function api(path, opts) {
    const res = await fetch(API_BASE + path, opts || {});
    const ct = res.headers.get("content-type") || "";
    let body = null;
    if (ct.includes("application/json")) {
      body = await res.json().catch(() => null);
    } else {
      body = await res.text();
    }
    if (!res.ok) {
      const msg = (body && (body.detail || body.error || body.message)) || (typeof body === "string" ? body : res.statusText) || `HTTP ${res.status}`;
      const err = new Error(msg);
      err.status = res.status;
      err.body = body;
      throw err;
    }
    return body;
  }

  function fmtDuration(ms) {
    if (!ms || ms < 0) ms = 0;
    const s = Math.floor(ms / 1000);
    const h = String(Math.floor(s / 3600)).padStart(2, "0");
    const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
    const ss = String(s % 60).padStart(2, "0");
    return `${h}:${m}:${ss}`;
  }

  function fmtKB(kb) {
    if (kb == null) return "";
    if (kb < 1024) return `${kb.toFixed(0)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  }

  function fmtDate(ts) {
    if (!ts) return "";
    try {
      const d = new Date(ts * 1000);
      return d.toLocaleString(undefined, { hour12: false });
    } catch (e) { return ""; }
  }

  function samePath(a, b) {
    if (!a || !b) return false;
    const norm = (s) => String(s).trim().toLowerCase().replace(/\\/g, "/").replace(/^[./]+/, "");
    const na = norm(a), nb = norm(b);
    return na === nb || na.endsWith("/" + nb) || nb.endsWith("/" + na);
  }

  function classifyLine(line) {
    if (!line) return "";
    const ll = line.toLowerCase();
    if (ll.includes("error") || ll.includes("traceback") || ll.includes("❌") || line.includes("ERROR")) return "log-line-err";
    if (ll.includes("warning") || ll.includes("warn") || ll.includes("⚠️")) return "log-line-warn";
    if (ll.startsWith("[visionsync] training finished") || ll.includes("🎉") || ll.includes("✅")) return "log-line-ok";
    if (line.startsWith("[VisionSync]") || ll.includes("epoch") || ll.includes("starting training")) return "log-line-info";
    return "";
  }

  function appendLogs(lines) {
    const frag = document.createDocumentFragment();
    for (const line of lines) {
      const div = document.createElement("div");
      const cls = classifyLine(line);
      if (cls) div.className = cls;
      div.textContent = line;
      frag.appendChild(div);
    }
    logBox.appendChild(frag);
    if (tailLogsCb.checked) {
      logBox.scrollTop = logBox.scrollHeight;
    }
  }

  async function refreshModels() {
    modelsListEl.innerHTML = '<div class="muted">Loading models...</div>';
    try {
      const data = await api("/training/models");
      const models = data.models || [];
      const active = data.active_model || "";

      // Populate Master base-model select too
      const prevSel = mBaseModel.value;
      mBaseModel.innerHTML = "";
      if (!models.length) {
        const o = document.createElement("option");
        o.value = "";
        o.textContent = "— Model မရှိသေးပါ။ Default yolov8n.pt ကိုရွေး —";
        mBaseModel.appendChild(o);
      }
      const defs = [
        ["yolov8n.pt", "yolov8n.pt (nano, default)"],
        ["yolov8s.pt", "yolov8s.pt (small)"],
        ["yolov8m.pt", "yolov8m.pt (medium)"],
      ];
      defs.forEach(([v, t]) => {
        const o = document.createElement("option");
        o.value = v;
        o.textContent = t;
        if (!models.length && v === "yolov8n.pt") o.selected = true;
        mBaseModel.appendChild(o);
      });
      models.forEach((m) => {
        const o = document.createElement("option");
        o.value = m.path;
        const aMatch = samePath(active, m.path) || samePath(active, m.name);
        o.textContent = `${m.name} · ${fmtKB(m.size_kb)}${aMatch ? " · ACTIVE" : ""}`;
        mBaseModel.appendChild(o);
      });
      if (prevSel) {
        const has = Array.from(mBaseModel.options).some(o => o.value === prevSel);
        if (has) mBaseModel.value = prevSel;
      }

      if (!models.length) {
        modelsListEl.innerHTML = '<div class="muted">Trained model မရှိသေးပါ။</div>';
        return;
      }
      modelsListEl.innerHTML = "";
      models.forEach((m) => {
        const isActive = active && (samePath(m.path, active) || samePath(m.name, active));
        const row = document.createElement("div");
        row.className = "model-item" + (isActive ? " active" : "");
        row.innerHTML = `
          <div class="model-info">
            <div class="model-name">
              ${escapeHtml(m.name)}
              ${isActive ? '<span class="active-chip">ACTIVE</span>' : ""}
            </div>
            <div class="model-path">${escapeHtml(m.path)}</div>
            <div class="model-meta">
              <span>${fmtKB(m.size_kb)}</span>
              <span>${fmtDate(m.modified)}</span>
            </div>
          </div>
          <button class="btn btn-tiny ${isActive ? "btn-success" : ""}" ${isActive ? "disabled" : ""} data-path="${escapeAttr(m.path)}">
            ${isActive ? "လက်ရှိသုံးနေသည်" : "Activate လုပ်မည်"}
          </button>
        `;
        const btn = row.querySelector("button");
        if (!isActive) {
          btn.addEventListener("click", async () => {
            if (!confirm(`"${m.name}" ကို active model အဖြစ်သတ်မှတ်မှာသေချာပါသလား?`)) return;
            try {
              btn.disabled = true;
              const r = await api("/training/activate-model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_path: m.path }),
              });
              setMsg(mUploadMsg, r.message || "Activate ဖြစ်သွားပြီ။", r.ok !== false ? "ok" : "err");
              await refreshModels();
              setTimeout(refreshModels, 500);
              setTimeout(refreshModels, 1500);
            } catch (e) {
              setMsg(mUploadMsg, "Activate မအောင်မြင် — " + e.message, "err");
            } finally {
              btn.disabled = false;
            }
          });
        }
        modelsListEl.appendChild(row);
      });
    } catch (e) {
      modelsListEl.innerHTML = `<div class="muted">Failed to load models: ${escapeHtml(e.message)}</div>`;
    }
  }

  function onPickMasterZip(e) {
    const f = e.target.files && e.target.files[0];
    if (f) {
      mZipFileName.textContent = f.name + " · " + fmtKB(f.size / 1024);
      mFilePicker.classList.add("has-file");
      if (!mDatasetName.value) {
        mDatasetName.value = smartDatasetNameFromZip(f.name);
        if (typeof mDatasetName.setSelectionRange === "function") {
          mDatasetName.title = mDatasetName.value;
        }
      }
      if (!mClassName.value) {
        mClassName.value = smartClassNameFromZip(f.name);
        if (typeof mClassName.setSelectionRange === "function") {
          mClassName.title = mClassName.value;
        }
      }
    } else {
      mZipFileName.textContent = "ပစ္စည်းအသစ် ZIP ကိုရွေးပါ";
      mFilePicker.classList.remove("has-file");
    }
  }

  /**
   * ZIP နာမည်ကနေ Dataset folder အမည် အတွက် အကောင်းဆုံးအရှည် (max 32) နဲ့သန့်ရှင်းပေးသည်။
   * Roboflow export နာမည်ပုံစံ အများစုကို သက်တောင့်သက်သာ ချုံ့ပေးသည်။
   */
  function smartDatasetNameFromZip(zipName) {
    let base = (zipName || "").replace(/\.zip$/i, "");
    base = base.replace(/[^a-zA-Z0-9-_]+/g, "_");
    base = base.replace(/__+/g, "_").replace(/^_|_$/g, "");
    if (base.length <= 32) return base;
    // အရင်ဆုံး Roboflow/YoloV8 ရဲ့ ထပ်တူတဲ့အဆုံးများ ဖြုတ်ကြည့်
    base = base
      .replace(/_yolov8$/i, "")
      .replace(/_yolov[0-9]+$/i, "")
      .replace(/_voc$/i, "")
      .replace(/_coco$/i, "")
      .replace(/_raw[-_ ]?images?/i, "_img")
      .replace(/_images?[-_ ]?raw/i, "_img")
      .replace(/_label(?:me|ing)?s?/i, "_lbl")
      .replace(/[-_ ]+v(\d+)/i, "_v$1");
    base = base.replace(/__+/g, "_").replace(/^_|_$/g, "");
    if (base.length <= 32) return base;
    // တောင်းဆိုသလောက် ရှည်နေဆဲ ဆိုရင် အစကိုယူပြီး နောက်ဆုံးမှာ _ (hash အနည်းငယ်) ထည့်
    return base.slice(0, 30).replace(/^_|_$/g, "") + "_" + base.slice(-2);
  }

  /**
   * ZIP နာမည်ကနေ Class အမည်အသစ်ကို အဓိပ္ပါယ်ရှိအောင် guess လုပ်ပေးသည်။
   * - Roboflow: "Project Name.v1-raw-images-classname.yolov8.zip" မျိုးဆိုရင် classname ကိုယူ
   * - အခြား pattern များ: yolov8, raw, images, labels, valid, voc, vN စသော suffix ဖြုတ်
   * - နောက်ဆုံးတွင် အကောင်းဆုံးစကားလုံး (တစ်ခု) သို့မဟုတ် နှစ်ခုအထိ ပြန်ပေးသည်။
   */
  function smartClassNameFromZip(zipName) {
    const s = (zipName || "").replace(/\.zip$/i, "");

    // Pattern 1 — Roboflow: "...className.yolov8", "...className_yolov8"
    // e.g. "Chicken Detection and Tracking.v1-raw-images-chickenlabel.yolov8"
    //        ထဲမှ "chickenlabel" → "chicken" ဆွဲထုတ်ဖို့ စမ်း
    let m;
    const reRoboflow = /[-_. ]([a-zA-Z]{3,})[\s._-]*(?:yolov\d+|voc|coco)(?:[\s._-]|$)/i;
    m = s.match(reRoboflow);
    if (m) {
      let cls = cleanupCandidate(m[1]);
      if (cls) return maybeStripLabelSuffix(cls);
    }

    // Pattern 2 — "projectname-classname-suffix" မျိုး အဆုံးမှာ yolov8 မပါတဲ့အခါ
    // နောက်ဆုံးသက်သက်သာစွာ ဖြုတ်ပြီးနောက်ဆုံးစကားလုံး ၁-၂ ခု
    let base = s.replace(/[^a-zA-Z0-9]+/g, "_");
    base = base
      .replace(/^_+|_+$/g, "")
      .replace(/_yolov\d+/gi, "")
      .replace(/_voc/gi, "")
      .replace(/_coco/gi, "")
      .replace(/_raw/gi, "")
      .replace(/_images?/gi, "")
      .replace(/_labels?/gi, "")
      .replace(/_train/gi, "")
      .replace(/_valid?/gi, "")
      .replace(/_test/gi, "")
      .replace(/_v\d+/gi, "")
      .replace(/_data$/i, "")
      .replace(/_dataset$/i, "")
      .replace(/_+$/g, "");

    const parts = base.split(/_+/).filter(Boolean);
    if (parts.length === 0) return "new_class";

    // နောက်ဆုံးစကားလုံးကို စစ် — "label", "labels", "labeled", "labelled", "classname" စတာ ဖြုတ်
    let last = parts[parts.length - 1];
    last = maybeStripLabelSuffix(last);

    // လုံးဝ "label" ဆိုတဲ့စကားလုံးပဲကျန်တယ်ဆိုရင် အရင်စကားလုံးပါယူ
    if (!last || ["label", "labels", "image", "images", "data", "dataset", "v"].includes(last.toLowerCase())) {
      if (parts.length >= 2) {
        const prev = parts[parts.length - 2];
        if (prev && prev.length >= 3) {
          return maybeStripLabelSuffix(prev.toLowerCase());
        }
      }
      // fallback: ပထမ meaningful စကားလုံး
      for (const p of parts) {
        if (p.length >= 3 && /^[a-zA-Z]/.test(p)) return maybeStripLabelSuffix(p.toLowerCase());
      }
      return parts.join("_").slice(0, 20).toLowerCase();
    }

    // အကယ်လို့ နောက်ဆုံး ၂ ခု ပေါင်းရင် အဓိပ္ပါယ်ပိုရှိရင် (ဥပမာ "cat_dog")
    if (parts.length >= 2) {
      const prev = parts[parts.length - 2];
      if (
        prev.length >= 3 &&
        last.length <= 8 &&
        (last === "detection" || last === "detect" || last === "segment" ||
         last === "classifier" || last === "classification" || last === "tracker" ||
         last === "tracking")
      ) {
        return maybeStripLabelSuffix(prev.toLowerCase()) + "_" + last;
      }
    }

    return last.toLowerCase();
  }

  function cleanupCandidate(x) {
    x = (x || "").replace(/[^a-zA-Z0-9]/g, "").trim();
    return x.length >= 2 ? x : null;
  }

  function maybeStripLabelSuffix(x) {
    const s = (x || "").toLowerCase();
    const strips = [
      "labelled", "labeled", "labelling", "labeling",
      "labels", "label", "lbl",
      "classname", "class", "cls",
      "name", "target", "object",
    ];
    for (const suf of strips) {
      if (s !== suf && s.endsWith(suf) && (s.length - suf.length) >= 2) {
        return s.slice(0, -suf.length);
      }
    }
    return s;
  }

  function dragDrop(picker, input, onChangeFn) {
    ["dragenter", "dragover"].forEach((ev) =>
      picker.addEventListener(ev, (e) => { e.preventDefault(); picker.style.borderColor = "var(--accent)"; })
    );
    ["dragleave", "drop"].forEach((ev) =>
      picker.addEventListener(ev, (e) => { e.preventDefault(); picker.style.borderColor = ""; })
    );
    picker.addEventListener("drop", (e) => {
      const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
      if (f) {
        const dt = new DataTransfer();
        dt.items.add(f);
        input.files = dt.files;
        onChangeFn({ target: input });
      }
    });
  }

  // ---------------------------------------------------------------------
  // Master tab — Tab switching, Upload+Merge, Master info refresh,
  //              start fine-tune, polling, activate+download
  // ---------------------------------------------------------------------
  function switchTab(name) {
    document.querySelectorAll(".tab").forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on ? "true" : "false");
    });
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.hidden = p.dataset.panel !== name;
    });
  }

  async function loadMasterInfo() {
    try {
      const info = await api("/master/info");
      mMasterBadge.textContent = `nc=${info.nc}`;
      mMasterBadge.classList.remove("tab-badge-dim");
      const tr = info.images_train || 0;
      const vl = info.images_val || 0;
      const names = info.names || [];
      const newest = names.length > 80 ? names.slice(80) : names.slice(Math.max(0, names.length - 15));

      let html = `
        <div class="m-master-info-grid">
          <div><span class="m-k">Total Classes (nc)</span><div class="m-v">${info.nc}</div></div>
          <div><span class="m-k">Images</span><div class="m-v">train ${tr} · val ${vl}</div></div>
          <div><span class="m-k">YAML Path</span><div class="m-v">${escapeHtml(info.yaml_path)}</div></div>
          <div><span class="m-k">Master Dir</span><div class="m-v">${escapeHtml(info.master_dir)}</div></div>
        </div>
        <div class="m-master-class-tags">
      `;
      if (names.length > 80) {
        html += `<span class="m-class-chip">0-79 · COCO 80 မျိုး</span>`;
      }
      newest.forEach((c) => {
        const isNew = c.id >= 80;
        html += `<span class="m-class-chip ${isNew ? "m-new" : ""}">${c.id}: ${escapeHtml(c.name)}</span>`;
      });
      html += `</div>`;
      mMasterInfo.innerHTML = html;
    } catch (e) {
      mMasterInfo.textContent = "Master info မရနိုင်ပါ — " + e.message;
      mMasterBadge.textContent = "error";
      mMasterBadge.classList.remove("tab-badge-dim");
    }
  }

  async function onMasterUpload(e) {
    e.preventDefault();
    const f = mZipFileInput.files && mZipFileInput.files[0];
    if (!f) { setMsg(mUploadMsg, "ZIP ဖိုင်ကိုရွေးပါ", "err"); return; }
    if (!mClassName.value.trim()) { setMsg(mUploadMsg, "Class နာမည်အသစ် ထည့်ပါ", "err"); return; }

    const fd = new FormData();
    fd.append("file", f);
    fd.append("class_name", mClassName.value.trim());
    if (mDatasetName.value.trim()) fd.append("dataset_name", mDatasetName.value.trim());

    try {
      mUploadBtn.disabled = true;
      mUploadBtn.textContent = "Upload + Merge နေသည်...";
      setMsg(mUploadMsg, "ZIP Upload → Extract → Master ထဲ Merge လုပ်နေသည်...", "ok");
      const r = await api("/master/upload-and-merge", { method: "POST", body: fd });
      setMsg(mUploadMsg, (r.ok ? "✅ အောင်မြင်ပါပြီ — " : "❌ မအောင်မြင် — ") + (r.message || ""), r.ok ? "ok" : "err");
      if (r.ok) {
        await loadMasterInfo();
        mZipFileInput.value = "";
        onPickMasterZip({ target: mZipFileInput });
      }
    } catch (e) {
      setMsg(mUploadMsg, "မအောင်မြင် — " + e.message, "err");
    } finally {
      mUploadBtn.disabled = false;
      mUploadBtn.textContent = "1. Upload + Merge → Master";
    }
  }

  function applyMasterStatus(st) {
    const s = st.status || "idle";
    const label = M_STATUS_LABELS[s] || s;
    let pct = 0;
    if (s === "ok") pct = 100;
    else if (s === "running") pct = Math.max(0, Math.min(100, Math.round(st.progress || 0)));
    mProgressFill.style.width = pct + "%";
    if (mProgressFill.parentElement && mProgressFill.parentElement.classList.contains("progress-bar")) {
      mProgressFill.parentElement.dataset.pct = String(pct);
    }
    mStatusLine.classList.remove("has-result", "has-error");
    if (st.started_at) mStartedAt = st.started_at;
    if (s === "running") {
      if (!mDurationTimer) {
        mDurationTimer = setInterval(() => {
          const end = st.finished_at ? st.finished_at : Date.now();
          mDurationText.textContent = fmtDuration(mStartedAt ? (end - mStartedAt) : 0);
        }, 500);
      }
      const epTxt = st.current_epoch && st.total_epochs ? ` · Epoch ${st.current_epoch}/${st.total_epochs}` : "";
      mStatusLine.textContent = (st.message || "Fine-Tuning လုပ်နေသည်။") + ` · ${pct}% ပြီးစီး${epTxt}`;
    } else {
      if (mDurationTimer) { clearInterval(mDurationTimer); mDurationTimer = null; }
      if (mStartedAt) {
        const end = st.finished_at ? st.finished_at : mStartedAt;
        mDurationText.textContent = fmtDuration(end - mStartedAt);
      } else {
        mDurationText.textContent = "00:00:00";
      }
      if (s === "ok") {
        mStatusLine.textContent = st.message || "အောင်မြင်ပြီးပြီ။";
        mStatusLine.classList.add("has-result");
      } else if (s === "error") {
        mStatusLine.textContent = "အမှား — " + (st.message || "Logs ကိုကြည့်ပါ");
        mStatusLine.classList.add("has-error");
      } else {
        mStatusLine.textContent = "အဆင့် (၁) Upload → (၂) Merge → (၃) Start လုပ်ပါ။";
      }
    }

    if (s === "ok" && st.best_pt) {
      mResultRow.hidden = false;
      mResultPt.textContent = st.best_pt;
      mResultPt.dataset.path = st.best_pt;
      if (st.archived_pt) {
        mResultPt.textContent += `  (backup: ${st.archived_pt})`;
        mResultPt.dataset.archived = st.archived_pt;
      }
    } else if (s === "error") {
      mResultRow.hidden = true;
    }
    // when idle, keep result visible if we finished
    if (s !== "running" && s !== "idle" && s !== "ok" && s !== "error") {
      mResultRow.hidden = true;
    }
  }

  async function pollMasterOnce() {
    try {
      const st = await api("/master/status");
      // New run detected (started_at changed) → reset log cursor
      if (st.started_at && st.started_at !== mLastStartedAt) {
        mLastStartedAt = st.started_at;
        mSince = 0;
      }
      // Incremental live logs → shared Live Logs panel
      if (Array.isArray(st.logs)) {
        if (st.logs.length > mSince) {
          appendLogs(st.logs.slice(mSince));
          mSince = st.logs.length;
        }
      }
      applyMasterStatus(st);
      if (st.status === "ok" || st.status === "error") {
        if (mPollTimer) { clearInterval(mPollTimer); mPollTimer = null; }
        refreshModels();
        loadMasterInfo();
      }
    } catch (e) {
      console.error(e);
    }
  }

  function startMasterPolling() {
    if (mPollTimer) return;
    mPollTimer = setInterval(pollMasterOnce, 2000);
    pollMasterOnce();
  }

  async function onMasterStart() {
    const base = mBaseModel.value.trim();
    if (!base) {
      setMsg(mUploadMsg, "Base .pt ဖိုင်ကို ရွေးပါ။ မရှိသေးရင် Models Refresh နှိပ်ပါ။", "err");
      return;
    }
    // We need a source folder for start-finetune endpoint. If user skipped direct upload,
    // we still require: user needs to at least have something merged.
    // We don't re-merge in start (that's upload-and-merge step). start-finetune expects
    // a source_root. Instead, modify to support skipping merge if master already has classes,
    // OR simply pass the last merged source_root. For simplicity: call start-finetune but
    // use a dummy empty (nonexistent) source_root causes 400. Instead add a small trick:
    // we first check master info. If nc>80 and we don't have a specific source folder,
    // it's fine, but the current API requires source_root. So we will make the start-finetune
    // API still work: we POST to start-finetune using master dataset directly via
    // a variant. For simplicity, re-use continuous_finetune directly through a new
    // endpoint `/master/start-finetune-only`? Instead, here's a pragmatic trick:
    // pass a new_source that is already inside the data. Or, we'll POST to an endpoint
    // that starts from the master yaml directly (no new source).
    //
    // So we add a "start from master directly" flow (no merge, just re-finetune) via
    // backend continuous_finetune() directly by calling a small wrapper API.

    const payload = {
      base_model: base,
      run_name: $("mRunName").value.trim() || "visionsync_master",
      epochs: Number($("mEpochs").value),
      imgsz: Number($("mImgsz").value),
      batch: Number($("mBatch").value),
      lr0: Number($("mLr0").value),
    };

    try {
      mStartBtn.disabled = true;
      setMsg(mUploadMsg, "Continuous Fine-Tuning စတင်နေသည်...", "ok");
      const r = await api("/master/start-direct-finetune", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setMsg(mUploadMsg, r.message || "Started", "ok");
      mResultRow.hidden = true;
      logBox.innerHTML = "";
      mSince = 0;
      mLastStartedAt = null;
      startMasterPolling();
    } catch (e) {
      setMsg(mUploadMsg, "Start မအောင်မြင် — " + e.message, "err");
      mStartBtn.disabled = false;
    }
  }

  async function onMasterActivate() {
    const p = mResultPt.dataset.archived || mResultPt.dataset.path;
    if (!p) return;
    if (!confirm(`\n${p}\n\nကို active အဖြစ်သတ်မှတ်မှာ သေချာပါသလား?`)) return;
    try {
      mActivateBtn.disabled = true;
      const r = await api("/training/activate-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_path: p }),
      });
      setMsg(mUploadMsg, r.message || "Activate ဖြစ်သွားပြီ။", "ok");
      await refreshModels();
      setTimeout(refreshModels, 600);
      setTimeout(refreshModels, 1800);
    } catch (e) {
      setMsg(mUploadMsg, "Activate မအောင်မြင် — " + e.message, "err");
    } finally {
      mActivateBtn.disabled = false;
    }
  }

  function onMasterDownload() {
    const p = mResultPt.dataset.archived || mResultPt.dataset.path;
    if (!p) return;
    const url = `/master/download-model?path=${encodeURIComponent(p)}`;
    const a = document.createElement("a");
    a.href = url;
    a.download = "";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function escapeHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, (m) => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;" }[m]));
  }
  function escapeAttr(s) { return escapeHtml(s); }

  function bind() {
    // Tab switching
    document.querySelectorAll(".tab").forEach((t) => {
      t.addEventListener("click", () => switchTab(t.dataset.tab));
    });

    $("refreshModels").addEventListener("click", refreshModels);
    $("clearLogsBtn").addEventListener("click", () => { logBox.innerHTML = ""; mSince = 0; });

    // Master
    mZipFileInput.addEventListener("change", onPickMasterZip);
    mUploadForm.addEventListener("submit", onMasterUpload);
    if (mFilePicker) dragDrop(mFilePicker, mZipFileInput, onPickMasterZip);
    mRefreshMasterBtn.addEventListener("click", loadMasterInfo);
    mStartBtn.addEventListener("click", onMasterStart);
    mActivateBtn.addEventListener("click", onMasterActivate);
    mDownloadBtn.addEventListener("click", onMasterDownload);
  }

  async function init() {
    bind();
    const mInit = await api("/master/status").catch(() => null);
    if (mInit) {
      applyMasterStatus(mInit);
      if (Array.isArray(mInit.logs) && mInit.logs.length) {
        appendLogs(mInit.logs);
        mSince = mInit.logs.length;
      }
      if (mInit.status === "running") startMasterPolling();
    }
    refreshModels();
    loadMasterInfo();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
