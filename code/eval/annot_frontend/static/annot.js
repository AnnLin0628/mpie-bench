(() => {
  const state = {
    meta: null,
    ann: null,
    split: "guide",
    items: [],
    idx: 0,
    qIndex: 0,
    answers: { overall: {}, inter: {}, anat: {} },
    t0: 0,
    focusIds: [],
  };

  const $ = (id) => document.getElementById(id);

  async function api(url, opts) {
    const r = await fetch(url, opts);
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || r.statusText || "request failed");
    return j;
  }

  function show(view) {
    $("view-home").classList.toggle("hidden", view !== "home");
    $("view-work").classList.toggle("hidden", view !== "work");
  }

  function renderHome() {
    const m = state.meta;
    $("guidelines-list").innerHTML = m.guidelines.map((g) => `<li>${esc(g)}</li>`).join("");
    $("guidelines-mini").innerHTML = m.guidelines.map((g) => `<li>${esc(g)}</li>`).join("");
    $("split-stats").innerHTML =
      Object.entries(m.splits)
        .map(([k, n]) => `<li><b>${k}</b>：${n} strip</li>`)
        .join("") +
      `<li>total <b>${m.n_total}</b> Article (Each annotator needs to complete independently)</li>` +
      `<li>protocol <b>${esc(m.protocol)}</b> · ${esc(m.scheme || "")}</li>`;

    const grid = $("ann-cards");
    grid.innerHTML = "";
    for (const a of m.annotators) {
      const p = m.progress[a.id] || { done: 0, total: m.n_total };
      const pct = p.total ? Math.round((100 * p.done) / p.total) : 0;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "ann-card";
      btn.innerHTML = `
        <h3>${esc(a.label)}</h3>
        <div>fixed identity ID：<code>${esc(a.id)}</code></div>
        <div class="bar"><i style="width:${pct}%"></i></div>
        <div class="stat">schedule ${p.done} / ${p.total}${p.done >= p.total && p.total ? " · Completed" : ""}</div>
      `;
      btn.onclick = () => enterAnn(a.id);
      grid.appendChild(btn);
    }
  }

  async function enterAnn(annId) {
    state.ann = annId;
    $("who-label").textContent = annId;
    renderSplitTabs();
    show("work");
    await loadTodo();
  }

  function renderSplitTabs() {
    const box = $("split-tabs");
    box.innerHTML = "";
    for (const s of ["guide", "pilot", "holdout"]) {
      const n = state.meta.splits[s] || 0;
      if (!n) continue;
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = `${s} (${n})`;
      b.classList.toggle("active", state.split === s);
      b.onclick = async () => {
        state.split = s;
        renderSplitTabs();
        await loadTodo();
      };
      box.appendChild(b);
    }
  }

  async function loadTodo() {
    const data = await api(
      `/api/items?ann=${encodeURIComponent(state.ann)}&split=${state.split}&todo=1`
    );
    state.items = data.items;
    state.idx = 0;
    const prog = await api(`/api/progress/${encodeURIComponent(state.ann)}`);
    $("prog-text").textContent = `total progress ${prog.done}/${prog.total} · This subset is awaiting bids ${state.items.length}`;
    if (!state.items.length) {
      $("empty-todo").classList.remove("hidden");
      $("stage-body").classList.add("hidden");
      return;
    }
    $("empty-todo").classList.add("hidden");
    $("stage-body").classList.remove("hidden");
    renderItem();
  }

  function applicableIds() {
    const ids = [];
    for (const q of state.meta.overall_questions || []) ids.push(["overall", q.id]);
    // Inter Fully open: do not press the system intent Hidden question
    for (const q of state.meta.inter_questions) ids.push(["inter", q.id]);
    for (const q of state.meta.anat_questions) ids.push(["anat", q.id]);
    return ids;
  }

  function setIntentHuman(v) {
    if (!v) return;
    state.intentHuman = v;
    const box = $("intent-human-box");
    if (!box) return;
    box.querySelectorAll("button").forEach((b) => {
      b.classList.toggle("on", b.dataset.v === v);
    });
  }

  function bindIntentButtons() {
    const box = $("intent-human-box");
    if (!box || box.dataset.bound === "1") return;
    box.dataset.bound = "1";
    // Event Delegation: Avoid Getting Problems .seg Logic mistying / Washed out by re-rendering
    box.addEventListener("click", (e) => {
      const btn = e.target && e.target.closest("button[data-v]");
      if (!btn || !box.contains(btn)) return;
      e.preventDefault();
      e.stopPropagation();
      setIntentHuman(btn.dataset.v);
    });
  }

  function renderItem() {
    const it = state.items[state.idx];
    if (!it) return;
    state.t0 = performance.now();
    state.answers = { overall: {}, inter: {}, anat: {} };
    state.qIndex = 0;
    state.intentSystem = it.intent || "unspecified";
    setIntentHuman(state.intentSystem);
    $("gen-img").src = it.img_url;
    $("meta-idx").textContent = `${state.idx + 1} / ${state.items.length}`;
    $("meta-ids").textContent = `${it.sample_id} · ${it.model_id}`;
    $("meta-intent").textContent = `System guess=${it.intent}`;
    $("meta-cat").textContent = it.cat || "";
    $("prompt-box").textContent =
      it.prompt_zh || it.prompt || "(There is no Chinese translation yet. Please see the English version below or refresh it later.)";
    $("prompt-en-box").textContent = it.prompt_en || it.prompt || "(No original English text)";
    $("intent-hint").textContent = `Expected number of people ${it.n_expected ?? "?"} · Inter All questions must be answered`;
    const sysHint = $("intent-system-hint");
    if (sysHint) {
      sysHint.textContent = `system/Keyword guessing:${it.intent}(Not reliable, please follow the editing instructions to make your own choices; do not skip the following questions because of guessing)`;
    }
    $("notes").value = "";
    $("save-msg").textContent = "";
    $("save-msg").className = "msg";

    state.focusIds = applicableIds();
    $("overall-form").innerHTML = (state.meta.overall_questions || [])
      .map((q) => qrowHtml("overall", q))
      .join("");
    $("inter-form").innerHTML = state.meta.inter_questions
      .map((q) => qrowHtml("inter", q))
      .join("");
    $("anat-form").innerHTML = state.meta.anat_questions
      .map((q) => qrowHtml("anat", q))
      .join("");

    bindSegButtons();
    highlightQ();
  }

  function qrowHtml(group, q) {
    const cond = q.cond ? `<span class="cond">${esc(q.cond)}</span>` : "";
    const scale = q.scale || "bin";
    let buttons;
    if (scale === "likert5") {
      buttons = [1, 2, 3, 4, 5]
        .map((v) => `<button type="button" data-v="${v}" title="${v}">${v}</button>`)
        .join("");
    } else if (scale === "ordinal3") {
      const L = q.labels || {
        "2": "2 fit",
        "1": "1 Not posted",
        "0": "0 No contact",
      };
      buttons = `
        <button type="button" data-v="2" title="${esc(L["2"] || "2")}">${esc(L["2"] || "2")}</button>
        <button type="button" data-v="1" title="${esc(L["1"] || "1")}">${esc(L["1"] || "1")}</button>
        <button type="button" data-v="0" title="${esc(L["0"] || "0")}">${esc(L["0"] || "0")}</button>
        <button type="button" data-v="U" title="Unable to judge">U</button>`;
    } else {
      buttons = `
        <button type="button" data-v="1" title="This picture is normal">1 normal</button>
        <button type="button" data-v="0" title="There is something wrong with this item in this picture">0 There is a problem</button>
        <button type="button" data-v="U" title="Unable to judge">U</button>`;
    }
    return `<div class="qrow" data-group="${group}" data-id="${q.id}">
      <div class="qid">${q.id}</div>
      <div>${esc(q.text)}${cond}</div>
      <div class="seg" data-group="${group}" data-id="${q.id}" data-scale="${scale}">
        ${buttons}
      </div>
    </div>`;
  }

  function bindSegButtons() {
    // Only tie the question button; do not tie the "Contact Intention" row
    document.querySelectorAll(".seg[data-group] button").forEach((btn) => {
      btn.onclick = () => {
        const seg = btn.parentElement;
        const g = seg.dataset.group;
        const id = seg.dataset.id;
        setAnswer(g, id, btn.dataset.v);
        // Ic=0 → clear Ir focus expectation in UI
        if (g === "inter" && id === "Ic" && btn.dataset.v === "0") {
          state.answers.inter.Ir = null;
          const irSeg = document.querySelector(`.seg[data-group="inter"][data-id="Ir"]`);
          if (irSeg) {
            irSeg.querySelectorAll("button").forEach((b) => (b.className = ""));
            const row = document.querySelector(`.qrow[data-group="inter"][data-id="Ir"]`);
            if (row) {
              const na = row.querySelector(".na-dyn");
              if (!na) {
                const hint = document.createElement("div");
                hint.className = "na na-dyn";
                hint.textContent = "Ic=0 → Ir Not applicable (automatically when saving null）";
                row.appendChild(hint);
              }
            }
          }
        }
        const pos = state.focusIds.findIndex(([gg, ii]) => gg === g && ii === id);
        if (pos >= 0 && pos < state.focusIds.length - 1) state.qIndex = pos + 1;
        highlightQ();
      };
    });
  }

  function setAnswer(group, id, v) {
    if (!state.answers[group]) state.answers[group] = {};
    state.answers[group][id] = v;
    const seg = document.querySelector(`.seg[data-group="${group}"][data-id="${id}"]`);
    if (!seg) return;
    seg.querySelectorAll("button").forEach((b) => {
      b.className = b.dataset.v === String(v) ? `on-${v}` : "";
    });
  }

  function highlightQ() {
    document.querySelectorAll(".qrow").forEach((r) => r.classList.remove("active"));
    const cur = state.focusIds[state.qIndex];
    if (!cur) return;
    const [g, id] = cur;
    const row = document.querySelector(`.qrow[data-group="${g}"][data-id="${id}"]`);
    if (row) row.classList.add("active");
  }

  async function saveAndNext() {
    const it = state.items[state.idx];
    if (!it) return;
    const msg = $("save-msg");
    try {
      if (!state.intentHuman) {
        throw new Error("Please select "Contact Intent" first (required / forbidden / unspecified）");
      }
      // Ic=0 → Not required Ir
      const ic = state.answers.inter.Ic;
      if (String(ic) === "0") state.answers.inter.Ir = null;
      const payload = {
        annotator_id: state.ann,
        sample_id: it.sample_id,
        model_id: it.model_id,
        key: it.key,
        split: it.split,
        intent_system: state.intentSystem || it.intent,
        intent_shown: state.intentSystem || it.intent,
        intent_human: state.intentHuman,
        overall: state.answers.overall,
        inter: state.answers.inter,
        anat: state.answers.anat,
        notes: $("notes").value.trim(),
        seconds: Math.round((performance.now() - state.t0) / 1000),
      };
      const res = await api("/api/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      msg.textContent = "Saved";
      msg.className = "msg ok";
      state.items.splice(state.idx, 1);
      $("prog-text").textContent = `total progress ${res.progress.done}/${res.progress.total} · This subset is awaiting bids ${state.items.length}`;
      if (!state.items.length) {
        $("empty-todo").classList.remove("hidden");
        $("stage-body").classList.add("hidden");
        return;
      }
      if (state.idx >= state.items.length) state.idx = 0;
      renderItem();
    } catch (e) {
      msg.textContent = e.message || String(e);
      msg.className = "msg err";
    }
  }

  function skip() {
    if (!state.items.length) return;
    state.idx = (state.idx + 1) % state.items.length;
    renderItem();
  }

  function onKey(e) {
    if ($("view-work").classList.contains("hidden")) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    const k = e.key.toLowerCase();
    if (k === "s" || k === "enter") {
      e.preventDefault();
      saveAndNext();
      return;
    }
    if (k === "n") {
      e.preventDefault();
      skip();
      return;
    }
    if (k === "j") {
      state.qIndex = Math.min(state.focusIds.length - 1, state.qIndex + 1);
      highlightQ();
      return;
    }
    if (k === "k") {
      state.qIndex = Math.max(0, state.qIndex - 1);
      highlightQ();
      return;
    }
    if (k === "0" || k === "1" || k === "2" || k === "3" || k === "4" || k === "5" || k === "u") {
      const cur = state.focusIds[state.qIndex];
      if (!cur) return;
      const [g, id] = cur;
      const qlist =
        g === "overall"
          ? state.meta.overall_questions || []
          : g === "inter"
            ? state.meta.inter_questions
            : state.meta.anat_questions;
      const q = qlist.find((x) => x.id === id) || {};
      const scale = q.scale || "bin";
      if (scale === "likert5") {
        if (k === "u" || k === "0") return;
      } else if (scale === "ordinal3") {
        if (k === "3" || k === "4" || k === "5") return;
      } else {
        if (k === "2" || k === "3" || k === "4" || k === "5") return;
      }
      const v = k === "u" ? "U" : k;
      setAnswer(g, id, v);
      if (g === "inter" && id === "Ic" && v === "0") {
        state.answers.inter.Ir = null;
      }
      if (state.qIndex < state.focusIds.length - 1) state.qIndex += 1;
      highlightQ();
    }
  }

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function boot() {
    state.meta = await api("/api/meta");
    renderHome();
    bindIntentButtons(); // static DOM, just bind it once at startup
    $("btn-home").onclick = () => {
      show("home");
      api("/api/meta").then((m) => {
        state.meta = m;
        renderHome();
      });
    };
    $("btn-save").onclick = saveAndNext;
    $("btn-skip").onclick = skip;
    document.addEventListener("keydown", onKey);
  }

  boot().catch((e) => {
    document.body.innerHTML = `<pre style="padding:24px;color:#9b2c2c">Loading failed: ${esc(e.message)}</pre>`;
  });
})();
