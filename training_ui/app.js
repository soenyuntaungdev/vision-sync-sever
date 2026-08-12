(function () {
  "use strict";

  const API_BASE = "";
  const $ = (id) => document.getElementById(id);

  const statusBadge = $("statusBadge");
  const modelsListEl = $("modelsList");
  const logBox = $("logBox");
  const tailLogsCb = $("tailLogs");

  let mSince = 0;
  let mStartedAt = null;
  let mPollTimer = null;
  let mDurationTimer = null;
  let mLastStartedAt = null;

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
  const mReplaySource = $("mReplaySource");
  const mReplayPerClass = $("mReplayPerClass");
  const mReplayBtn = $("mReplayBtn");
  const mReplayRemoveBtn = $("mReplayRemoveBtn");
  const mReplayBadge = $("mReplayBadge");
  const mReplayMsg = $("mReplayMsg");
  const mBaseModel = $("mBaseModel");
  const mStartBtn = $("mStartBtn");
  const mProgressFill = $("mProgressFill");
  const mStatusLine = $("mStatusLine");
  const mDurationText = $("mDurationText");
  const mResultRow = $("mResultRow");
  const mResultPt = $("mResultPt");
  const mActivateBtn = $("mActivateBtn");
  const mDownloadBtn = $("mDownloadBtn");

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

  function setStatus(s) {
    statusBadge.textContent = M_STATUS_LABELS[s] || s;
    const cls = {
      idle: "badge-idle",
      running: "badge-running",
      ok: "badge-completed",
      error: "badge-failed",
    }[s] || "badge-idle";
    statusBadge.className = "badge " + cls;
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
        o.textContent = `${m.name} · ${fmtKB(m.size_kb)}${active === m.path || active === m.name ? " · ACTIVE" : ""}`;
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
        const isActive = active && (m.path === active || m.name === active);
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
            if (!confirm(`"${m.name}" ကို active model အဖြစ် သတ်မှတ်မှာ သေချာပါသလား?`)) return;
            try {
              btn.disabled = true;
              const r = await api("/training/activate-model", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_path: m.path }),
              });
              setMsg(mUploadMsg, r.message || "Activate ဖြစ်သွားပါပြီ။", "ok");
              setTimeout(refreshModels, 300);
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
  // Master — Master info refresh, Upload+Merge, start fine-tune,
  //          polling, activate+download
  // ---------------------------------------------------------------------
  // ---- Stepper ----
  const mStep = { replay: false, newClass: false, training: false, result: false };

  function updateStepper() {
    const states = [
      mStep.replay ? "is-done" : "is-active",
      mStep.newClass ? "is-done" : (mStep.replay ? "is-active" : ""),
      mStep.result ? "is-done" : (mStep.training ? "is-active" : (mStep.newClass ? "is-active" : "")),
      mStep.result ? "is-active" : "",
    ];
    document.querySelectorAll("#mStepper .step").forEach((el, i) => {
      el.classList.remove("is-done", "is-active");
      if (states[i]) el.classList.add(states[i]);
    });
  }

  async function loadMasterInfo() {
    try {
      const info = await api("/master/info");
      const audit = await api("/master/audit").catch(() => null);

      const tr = info.images_train || 0;
      const vl = info.images_val || 0;
      const names = info.names || [];
      const newClasses = names.filter((c) => c.id >= 80);
      const emptyCount = audit && Array.isArray(audit.empty_classes) ? audit.empty_classes.length : null;

      mStep.newClass = newClasses.length > 0;
      updateStepper();

      mMasterBadge.textContent = `nc=${info.nc}`;
      mMasterBadge.classList.remove("tab-badge-dim");

      // label မရှိသော class များ = fine-tune ပြီးရင် ပျောက်သွားမည့် class များ
      let emptyCls = "is-ok", emptyTxt = "—";
      if (emptyCount !== null) {
        emptyTxt = String(emptyCount);
        emptyCls = emptyCount === 0 ? "is-ok" : (emptyCount > 40 ? "is-warn" : "");
      }

      let html = `
        <div class="m-stat-strip">
          <div class="m-stat">
            <span class="m-stat-k">Total Classes</span>
            <span class="m-stat-v">${info.nc}
              <span class="m-stat-sub">${newClasses.length ? `+${newClasses.length} အသစ်` : ""}</span>
            </span>
          </div>
          <div class="m-stat">
            <span class="m-stat-k">Images</span>
            <span class="m-stat-v">${tr + vl}
              <span class="m-stat-sub">${tr}/${vl}</span>
            </span>
          </div>
          <div class="m-stat">
            <span class="m-stat-k">Label မရှိသော class</span>
            <span class="m-stat-v ${emptyCls}">${emptyTxt}</span>
          </div>
        </div>
      `;

      if (emptyCount !== null && emptyCount > 40) {
        html += `<p class="card-note"><b class="note-warn">သတိပြုရန်</b> — class ${emptyCount} ခုမှာ label မရှိပါ။
                 ဒီအတိုင်း train ရင် အဲဒီ class တွေ ပျောက်သွားပါမည်။ အဆင့် ၁ (COCO Replay) ကို အရင်လုပ်ပါ။</p>`;
      }

      html += `<div class="m-master-class-tags">`;
      if (names.length > 80) html += `<span class="m-class-chip">0–79 · COCO ၈၀ မျိုး</span>`;
      const shown = newClasses.length ? newClasses : names.slice(Math.max(0, names.length - 10));
      shown.forEach((c) => {
        const n = audit && audit.counts ? audit.counts[c.name] : undefined;
        const cnt = n === undefined ? "" : ` · ${n}`;
        html += `<span class="m-class-chip ${c.id >= 80 ? "m-new" : ""}">${c.id}: ${escapeHtml(c.name)}${cnt}</span>`;
      });
      html += `</div>`;
      html += `<div class="m-path-line" title="${escapeAttr(info.yaml_path)}">${escapeHtml(info.yaml_path)}</div>`;
      mMasterInfo.innerHTML = html;
    } catch (e) {
      mMasterInfo.textContent = "Master info မရနိုင်ပါ — " + e.message;
      mMasterBadge.textContent = "error";
      mMasterBadge.classList.remove("tab-badge-dim");
    }
  }

  // ---- COCO Replay (အဟောင်း ၈၀ မျိုး မပျောက်စေရန်) ----
  let mReplayPollTimer = null;

  function applyReplayState(st) {
    const info = st.info || {};
    if (st.status === "running") {
      mReplayBadge.textContent = "ထည့်နေသည်...";
      mReplayBadge.classList.remove("tab-badge-dim");
      setMsg(mReplayMsg, st.message || "လုပ်ဆောင်နေသည်...", "ok");
      mReplayBtn.disabled = true;
      return;
    }
    mReplayBtn.disabled = false;
    if (info.present) {
      mReplayBadge.textContent = `${info.source} · ${(info.images_train || 0) + (info.images_val || 0)} ပုံ`;
      mReplayBadge.classList.remove("tab-badge-dim");
    } else {
      mReplayBadge.textContent = "မထည့်ရသေး";
      mReplayBadge.classList.add("tab-badge-dim");
    }
    mStep.replay = !!info.present;
    updateStepper();
    if (st.status === "ok") setMsg(mReplayMsg, st.message || "ပြီးပါပြီ။", "ok");
    else if (st.status === "error") setMsg(mReplayMsg, "မအောင်မြင် — " + (st.message || ""), "err");
  }

  async function pollReplayOnce() {
    try {
      const st = await api("/master/replay-status");
      if (Array.isArray(st.progress_logs) && st.progress_logs.length) {
        setMsg(mReplayMsg, st.progress_logs[st.progress_logs.length - 1], "ok");
      }
      applyReplayState(st);
      if (st.status !== "running") {
        if (mReplayPollTimer) { clearInterval(mReplayPollTimer); mReplayPollTimer = null; }
        loadMasterInfo();
      }
    } catch (e) {
      console.error(e);
    }
  }

  async function onReplayAdd() {
    const source = mReplaySource.value;
    const per_class = Number(mReplayPerClass.value) || 30;
    const warn = source === "val2017"
      ? "\n\nval2017 က ~၈၀၀MB download လုပ်ပါမည် (ပထမတစ်ခါသာ)။"
      : "";
    if (!confirm(`COCO replay (${source}) ကို master ထဲ ထည့်မှာ သေချာပါသလား?${warn}`)) return;
    try {
      mReplayBtn.disabled = true;
      const r = await api("/master/add-coco-replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source, per_class }),
      });
      setMsg(mReplayMsg, r.message || "စတင်လိုက်ပါပြီ။", "ok");
      if (!mReplayPollTimer) mReplayPollTimer = setInterval(pollReplayOnce, 2000);
      pollReplayOnce();
    } catch (e) {
      setMsg(mReplayMsg, "မအောင်မြင် — " + e.message, "err");
      mReplayBtn.disabled = false;
    }
  }

  async function onReplayRemove() {
    if (!confirm("Master ထဲက COCO replay ပုံများကို ဖျက်မှာ သေချာပါသလား?")) return;
    try {
      const r = await api("/master/coco-replay", { method: "DELETE" });
      setMsg(mReplayMsg, r.message || "ဖျက်ပြီးပါပြီ။", "ok");
      await pollReplayOnce();
      await loadMasterInfo();
    } catch (e) {
      setMsg(mReplayMsg, "ဖျက်လို့မရပါ — " + e.message, "err");
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
    setStatus(s);
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

    mStep.training = s === "running";
    mStep.result = s === "ok" && !!st.best_pt;
    updateStepper();

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
      freeze: Number($("mFreeze").value),
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
      setMsg(mUploadMsg, r.message || "Activate ဖြစ်သွားပါပြီ။", "ok");
      setTimeout(refreshModels, 400);
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
    $("refreshModels").addEventListener("click", refreshModels);
    $("clearLogsBtn").addEventListener("click", () => { logBox.innerHTML = ""; mSince = 0; });

    mZipFileInput.addEventListener("change", onPickMasterZip);
    mUploadForm.addEventListener("submit", onMasterUpload);
    if (mFilePicker) dragDrop(mFilePicker, mZipFileInput, onPickMasterZip);
    mRefreshMasterBtn.addEventListener("click", loadMasterInfo);
    mReplayBtn.addEventListener("click", onReplayAdd);
    mReplayRemoveBtn.addEventListener("click", onReplayRemove);
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
    pollReplayOnce();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
