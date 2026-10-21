const $ = (id) => document.getElementById(id);
let currentTask = null, lastCompleted = 0, sessionId = null, pageIndex = 0, pagesMeta = [];
let pageState = { width: 0, height: 0, boxes: [], name: "" };
let selected = [], tool = "select", zoom = 1, pan = { x: 0, y: 0 };
let mergeFrom = null, undoStack = [], redoStack = [], undoKind = "", textUndoId = null;
let drag = null, world = null, imgEl = null;

function saveCfg(silent = false) {
  localStorage.setItem("jpmt109ollama", JSON.stringify({
    endpoint: $("endpoint").value, model: $("model").value, key: $("key").value,
    batch: $("batch").value, maxout: $("maxout").value, timeout: $("timeout").value,
  }));
  if (!silent) alert("已保存到本机浏览器");
}
(function () {
  try {
    const c = JSON.parse(localStorage.getItem("jpmt109ollama") || "{}");
    for (const k of ["endpoint", "model", "key", "batch", "maxout", "timeout"]) if (c[k] != null) $(k).value = c[k];
  } catch (e) {}
})();

async function testOCR() {
  const s = $("ollamastatus"); s.style.display = "block"; s.textContent = "正在测试 PP-OCRv6_manga…";
  try {
    const r = await fetch("/api/ocr_status"); const d = await r.json();
    if (!r.ok || !d.ok) throw Error(d.error || "OCR 不可用");
    s.textContent = "OCR 正常：" + d.model + "\n检测：" + d.detector + "\n识别：" + d.recognizer + "\n设备：" + d.provider;
  } catch (e) { s.textContent = "OCR 模型不可用：" + e.message; }
}
async function testOllama() {
  const s = $("ollamastatus"); s.style.display = "block"; s.textContent = "正在测试 Ollama…";
  try {
    const r = await fetch("/api/ollama_status"); const d = await r.json();
    if (!r.ok || !d.ok) throw Error(d.error || "Ollama 不可用");
    s.textContent = "Ollama 正常。模型：" + (d.models || []).join(", ") + "\nHy-MT2-7B：" + (d.has_hymt2 ? "已发现" : "未发现");
  } catch (e) { s.textContent = "Ollama 连接失败：" + e.message; }
}

async function startReview() {
  const f = $("ocrfile").files[0];
  if (!f) { alert("先选择图片或压缩包"); return; }
  $("ocrbtn").disabled = true;
  $("ocrstatus").textContent = "正在上传并 OCR…";
  const fd = new FormData(); fd.append("file", f);
  try {
    const r = await fetch("/api/review_start", { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || !d.ok) throw Error(d.error || "OCR 失败");
    sessionId = d.session_id;
    pagesMeta = Array.from({ length: d.total }, (_, i) => ({ i, confirmed: false }));
    $("editorWrap").style.display = "block";
    await pollReview();
  } catch (e) {
    $("ocrstatus").textContent = "失败：" + e.message;
  } finally { $("ocrbtn").disabled = false; }
}

async function pollReview() {
  if (!sessionId) return;
  const r = await fetch("/api/review_status/" + sessionId + "?t=" + Date.now(), { cache: "no-store" });
  const d = await r.json();
  if (!r.ok || !d.ok) throw Error(d.error || "读取失败");
  $("ocrstatus").textContent = d.message || "";
  if (d.status === "error") throw Error(d.error || d.message || "OCR 失败");
  if (d.status === "ocr") { setTimeout(pollReview, 700); return; }
  pageIndex = 0;
  await loadPage(0);
}

async function loadPage(i) {
  const r = await fetch(`/api/review_page/${sessionId}/${i}`);
  const d = await r.json();
  if (!r.ok || !d.ok) throw Error(d.error || "读页失败");
  pageIndex = i;
  pageState = { width: d.width, height: d.height, boxes: d.boxes || [], name: d.name, confirmed: d.confirmed };
  selected = [];
  mergeFrom = null; undoStack = []; redoStack = []; undoKind = ""; textUndoId = null;
  $("pageMeta").textContent = `${i + 1}/${pagesMeta.length}  ${d.name}  ${pageState.boxes.length} 框`;
  renderFilm();
  mountStage(d.image_url + "?t=" + Date.now());
  syncSide();
}

function renderFilm() {
  const film = $("film"); film.innerHTML = "";
  pagesMeta.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = i === pageIndex ? "on" : "";
    const im = document.createElement("img");
    im.src = `/api/review_image/${sessionId}/${i}`;
    b.appendChild(im);
    b.onclick = () => { savePage(false).then(() => loadPage(i)); };
    film.appendChild(b);
  });
}

function mountStage(src) {
  const stage = $("stage");
  stage.innerHTML = "";
  world = document.createElement("div"); world.className = "world";
  imgEl = document.createElement("img"); imgEl.src = src;
  imgEl.onload = () => {
    pageState.width = imgEl.naturalWidth; pageState.height = imgEl.naturalHeight;
    world.style.width = pageState.width + "px";
    world.style.height = pageState.height + "px";
    fit(); drawBoxes();
  };
  world.appendChild(imgEl);
  stage.appendChild(world);
  stage.onwheel = (e) => {
    e.preventDefault();
    zoomAt(e.clientX, e.clientY, e.deltaY > 0 ? 0.9 : 1.12);
  };
  stage.oncontextmenu = (e) => e.preventDefault();
  stage.onpointerdown = onDown;
  stage.onpointermove = onMove;
  stage.onpointerup = onUp;
  stage.onpointercancel = onUp;
}

function fit() {
  const stage = $("stage");
  const pad = 24;
  const sx = (stage.clientWidth - pad) / pageState.width;
  const sy = (stage.clientHeight - pad) / pageState.height;
  zoom = Math.max(0.12, Math.min(4, Math.min(sx, sy)));
  pan = {
    x: (stage.clientWidth - pageState.width * zoom) / 2,
    y: (stage.clientHeight - pageState.height * zoom) / 2,
  };
  applyView();
}
function applyView() {
  if (!world) return;
  world.style.left = "0px";
  world.style.top = "0px";
  world.style.transformOrigin = "0 0";
  world.style.transform = `translate(${pan.x}px,${pan.y}px) scale(${zoom})`;
}
function zoomAt(clientX, clientY, factor) {
  const rect = $("stage").getBoundingClientRect();
  const mx = clientX - rect.left, my = clientY - rect.top;
  const ix = (mx - pan.x) / zoom, iy = (my - pan.y) / zoom;
  const nz = Math.max(0.12, Math.min(8, zoom * factor));
  zoom = nz;
  pan = { x: mx - ix * nz, y: my - iy * nz };
  applyView();
}

function toImg(e) {
  const rect = $("stage").getBoundingClientRect();
  return {
    x: (e.clientX - rect.left - pan.x) / zoom,
    y: (e.clientY - rect.top - pan.y) / zoom,
  };
}

function hitBox(p) {
  for (let i = pageState.boxes.length - 1; i >= 0; i--) {
    const b = pageState.boxes[i].box;
    if (p.x >= b[0] && p.x <= b[2] && p.y >= b[1] && p.y <= b[3]) return pageState.boxes[i];
  }
  return null;
}

function onDown(e) {
  if (e.button === 2 || e.button === 1) {
    e.preventDefault();
    try { $("stage").setPointerCapture(e.pointerId); } catch (err) {}
    drag = { type: "pan", x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
    $("stage").classList.add("pan");
    return;
  }
  if (e.button !== 0) return;
  try { $("stage").setPointerCapture(e.pointerId); } catch (err) {}
  const p = toImg(e);
  if (tool === "draw") { drag = { type: "draw", x: p.x, y: p.y }; return; }
  const box = hitBox(p);
  if (box) {
    if (mergeFrom != null) {
      if (mergeFrom < 0 || mergeFrom === box.id) {
        mergeFrom = box.id;
        selected = [box.id];
      } else {
        const first = pageState.boxes.find((b) => b.id === mergeFrom);
        if (first) mergeTwo(first, box);
      }
      drawBoxes(); syncSide(); return;
    }
    if (e.shiftKey) {
      selected = selected.includes(box.id) ? selected.filter((id) => id !== box.id) : selected.concat(box.id);
      drawBoxes(); syncSide(); return;
    }
    if (!selected.includes(box.id)) selected = [box.id];
    const h = handleAt(box, p);
    if (h) drag = { type: "resize", id: box.id, handle: h, x: p.x, y: p.y, orig: box.box.slice(), snapped: false };
    else {
      const orig = {};
      pageState.boxes.forEach((b) => { if (selected.includes(b.id)) orig[b.id] = b.box.slice(); });
      drag = { type: "move", x: p.x, y: p.y, orig, snapped: false };
    }
    drawBoxes(); syncSide(); return;
  }
  if (mergeFrom != null) return;
  if (!e.shiftKey) selected = [];
  drawBoxes(); syncSide();
}

function handleAt(box, p) {
  const [x1, y1, x2, y2] = box.box;
  const pts = { nw: [x1, y1], n: [(x1 + x2) / 2, y1], ne: [x2, y1], e: [x2, (y1 + y2) / 2], se: [x2, y2], s: [(x1 + x2) / 2, y2], sw: [x1, y2], w: [x1, (y1 + y2) / 2] };
  const t = Math.max(12, 18 / zoom);
  for (const k of Object.keys(pts)) if (Math.abs(p.x - pts[k][0]) <= t && Math.abs(p.y - pts[k][1]) <= t) return k;
  return null;
}

function onMove(e) {
  if (!drag) return;
  if (drag.type === "pan") { pan = { x: drag.px + (e.clientX - drag.x), y: drag.py + (e.clientY - drag.y) }; applyView(); return; }
  const p = toImg(e);
  if (drag.type === "draw") {
    drag.cur = [Math.min(drag.x, p.x), Math.min(drag.y, p.y), Math.max(drag.x, p.x), Math.max(drag.y, p.y)];
    drawBoxes(); return;
  }
  if (!drag.snapped) { drag.snapped = true; pushUndo("geom"); }
  if (drag.type === "move") {
    const dx = p.x - drag.x, dy = p.y - drag.y;
    pageState.boxes.forEach((b) => {
      if (!drag.orig[b.id]) return;
      const o = drag.orig[b.id];
      b.box = clip([o[0] + dx, o[1] + dy, o[2] + dx, o[3] + dy]);
      if (b.member_boxes && b.member_boxes.length === 1) b.member_boxes = [b.box.slice()];
      b.locked = true;
    });
    drawBoxes(); return;
  }
  if (drag.type === "resize") {
    const b = pageState.boxes.find((x) => x.id === drag.id); if (!b) return;
    b.box = clip(resize(drag.orig, drag.handle, p.x - drag.x, p.y - drag.y));
    if (b.member_boxes && b.member_boxes.length === 1) b.member_boxes = [b.box.slice()];
    b.locked = true; drawBoxes();
  }
}

function onUp() {
  if (drag && drag.type === "draw" && drag.cur) {
    const b = clip(drag.cur);
    if (b[2] - b[0] >= 12 && b[3] - b[1] >= 12) {
      pushUndo("draw");
      const id = (pageState.boxes.reduce((m, x) => Math.max(m, x.id), 0) || 0) + 1;
      const kind = autoKind(b);
      pageState.boxes.push({ id, box: b, text: "", kind, locked: true, member_boxes: [b.slice()] });
      selected = [id];
      recognizeBox(b).then((t) => { const x = pageState.boxes.find((q) => q.id === id); if (x) { x.text = t; syncSide(); } });
    }
  }
  drag = null; drawBoxes(); syncSide();
}

function clip(b) {
  const x1 = Math.max(0, Math.min(pageState.width, Math.round(b[0])));
  const y1 = Math.max(0, Math.min(pageState.height, Math.round(b[1])));
  const x2 = Math.max(0, Math.min(pageState.width, Math.round(b[2])));
  const y2 = Math.max(0, Math.min(pageState.height, Math.round(b[3])));
  return [Math.min(x1, x2), Math.min(y1, y2), Math.max(x1, x2), Math.max(y1, y2)];
}
function resize(o, h, dx, dy) {
  let [x1, y1, x2, y2] = o.slice();
  if (h.includes("w")) x1 += dx; if (h.includes("e")) x2 += dx;
  if (h.includes("n")) y1 += dy; if (h.includes("s")) y2 += dy;
  if (x2 - x1 < 12) { if (h.includes("w")) x1 = x2 - 12; else x2 = x1 + 12; }
  if (y2 - y1 < 12) { if (h.includes("n")) y1 = y2 - 12; else y2 = y1 + 12; }
  return [x1, y1, x2, y2];
}
function autoKind(b) {
  const w = b[2] - b[0], h = b[3] - b[1];
  if (w >= pageState.width * 0.38 || (w * h >= pageState.width * pageState.height * 0.1 && w >= h * 0.85)) return "long";
  return "bubble";
}

function drawBoxes() {
  if (!world || !imgEl) return;
  [...world.querySelectorAll(".bbox,.draft")].forEach((n) => n.remove());
  pageState.boxes.forEach((b) => {
    const el = document.createElement("div");
    el.className = "bbox" + (b.kind === "long" ? " long" : "") + (selected.includes(b.id) || mergeFrom === b.id ? " on" : "");
    el.style.left = b.box[0] + "px"; el.style.top = b.box[1] + "px";
    el.style.width = (b.box[2] - b.box[0]) + "px"; el.style.height = (b.box[3] - b.box[1]) + "px";
    const tag = document.createElement("span"); tag.className = "tag"; tag.textContent = b.id + (b.kind === "long" ? " 长" : "");
    el.appendChild(tag);
    if (selected.includes(b.id) && mergeFrom == null) {
      const w = b.box[2] - b.box[0], h = b.box[3] - b.box[1];
      [["nw", -5, -5], ["n", w / 2 - 5, -5], ["ne", w - 5, -5], ["e", w - 5, h / 2 - 5], ["se", w - 5, h - 5], ["s", w / 2 - 5, h - 5], ["sw", -5, h - 5], ["w", -5, h / 2 - 5]].forEach(([k, x, y]) => {
        const hd = document.createElement("i"); hd.className = "handle"; hd.style.left = x + "px"; hd.style.top = y + "px"; el.appendChild(hd);
      });
    }
    world.appendChild(el);
  });
  if (drag && drag.type === "draw" && drag.cur) {
    const el = document.createElement("div"); el.className = "bbox draft";
    el.style.left = drag.cur[0] + "px"; el.style.top = drag.cur[1] + "px";
    el.style.width = (drag.cur[2] - drag.cur[0]) + "px"; el.style.height = (drag.cur[3] - drag.cur[1]) + "px";
    el.style.borderStyle = "dashed"; el.style.borderColor = "#fff";
    world.appendChild(el);
  }
}

function primary() { return pageState.boxes.find((b) => b.id === selected[0]); }
function syncHist() {
  const u = $("undoBtn"); if (u) u.disabled = undoStack.length === 0;
  const r = $("redoBtn"); if (r) r.disabled = redoStack.length === 0;
  const mb = $("mergeBtn"); if (mb) mb.className = mergeFrom != null ? "" : "ghost";
  const h = $("mergeHint");
  if (!h) return;
  if (mergeFrom == null) h.textContent = "右键拖动移动画面；滚轮以光标为中心缩放；先点框再点合并再点第二个框。Ctrl+Z 撤回。";
  else if (mergeFrom < 0) h.textContent = "合并：先点第一个框，再点第二个框。Esc 取消。";
  else h.textContent = "已选 #" + mergeFrom + "，再点另一个框完成合并。Esc 取消。";
}
function pushUndo(kind, id) {
  if (kind === "text" && undoKind === "text" && textUndoId === id) return;
  undoStack.push(JSON.parse(JSON.stringify(pageState.boxes)));
  if (undoStack.length > 40) undoStack.shift();
  redoStack = [];
  undoKind = kind || "";
  textUndoId = kind === "text" ? id : null;
  syncHist();
}
function undoOnce() {
  if (!undoStack.length) return;
  redoStack.push(JSON.parse(JSON.stringify(pageState.boxes)));
  pageState.boxes = undoStack.pop();
  selected = selected.filter((id) => pageState.boxes.some((b) => b.id === id));
  if (mergeFrom != null && mergeFrom >= 0 && !pageState.boxes.some((b) => b.id === mergeFrom)) mergeFrom = null;
  undoKind = ""; textUndoId = null;
  drawBoxes(); syncSide();
}
function redoOnce() {
  if (!redoStack.length) return;
  undoStack.push(JSON.parse(JSON.stringify(pageState.boxes)));
  pageState.boxes = redoStack.pop();
  selected = selected.filter((id) => pageState.boxes.some((b) => b.id === id));
  undoKind = ""; textUndoId = null;
  drawBoxes(); syncSide();
}
function syncSide() {
  const p = primary();
  $("boxText").value = p ? p.text : "";
  $("kindBubble").className = p && p.kind !== "long" ? "" : "ghost";
  $("kindLong").className = p && p.kind === "long" ? "" : "ghost";
  $("toolSelect").className = tool === "select" ? "" : "ghost";
  $("toolDraw").className = tool === "draw" ? "" : "ghost";
  $("stage").classList.toggle("draw", tool === "draw");
  syncHist();
}
function setTool(t) { tool = t; if (t === "draw") mergeFrom = null; syncSide(); }
function setKind(k) {
  const p = primary(); if (!p) return;
  pushUndo("kind");
  p.kind = k; p.locked = true; drawBoxes(); syncSide();
}
$("boxText").addEventListener("input", () => {
  const p = primary(); if (!p) return;
  pushUndo("text", p.id);
  p.text = $("boxText").value; p.locked = true;
});

function mergeTwo(a, b) {
  pushUndo("merge");
  const box = [
    Math.min(a.box[0], b.box[0]), Math.min(a.box[1], b.box[1]),
    Math.max(a.box[2], b.box[2]), Math.max(a.box[3], b.box[3]),
  ];
  const members = [a, b].flatMap((x) => x.member_boxes || [x.box]);
  const text = [a, b].map((x) => (x.text || "").trim()).filter(Boolean).join("\n");
  const id = Math.min(a.id, b.id);
  const kind = a.kind === "long" && b.kind === "long" ? "long" : autoKind(box);
  pageState.boxes = pageState.boxes.filter((x) => x.id !== a.id && x.id !== b.id);
  pageState.boxes.push({ id, box, text, kind, locked: true, member_boxes: members });
  selected = [id]; mergeFrom = null;
  drawBoxes(); syncSide();
}
function mergeSelected() {
  const group = pageState.boxes.filter((b) => selected.includes(b.id));
  if (group.length >= 2) { mergeTwo(group[0], group[1]); return; }
  mergeFrom = group.length === 1 ? group[0].id : -1;
  drawBoxes(); syncSide();
}
function deleteSelected() {
  if (!selected.length) return;
  pushUndo("delete");
  pageState.boxes = pageState.boxes.filter((b) => !selected.includes(b.id));
  selected = []; drawBoxes(); syncSide();
}
async function recognizeBox(box) {
  const r = await fetch(`/api/review_recognize/${sessionId}/${pageIndex}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ box }),
  });
  const d = await r.json();
  return d.text || "";
}
async function recognizeSelected() {
  const p = primary(); if (!p) return;
  p.text = await recognizeBox(p.box); syncSide();
}
async function savePage(confirmed) {
  if (!sessionId) return;
  await fetch(`/api/review_save/${sessionId}/${pageIndex}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boxes: pageState.boxes, confirmed }),
  });
  if (pagesMeta[pageIndex]) pagesMeta[pageIndex].confirmed = !!confirmed;
}
async function lockPage() { await savePage(true); alert("本页已锁定"); }
async function lockAll() {
  await savePage(true);
  for (let i = 0; i < pagesMeta.length; i++) {
    if (i === pageIndex) continue;
    const r = await fetch(`/api/review_page/${sessionId}/${i}`);
    const d = await r.json();
    await fetch(`/api/review_save/${sessionId}/${i}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: d.boxes || [], confirmed: true }),
    });
  }
  alert("全部页面已按当前框锁定");
}
function prevPage() { if (pageIndex > 0) savePage(false).then(() => loadPage(pageIndex - 1)); }
function nextPage() { if (pageIndex < pagesMeta.length - 1) savePage(false).then(() => loadPage(pageIndex + 1)); }

window.addEventListener("keydown", (e) => {
  const typing = ["INPUT", "TEXTAREA"].includes((e.target || {}).tagName);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    if (e.shiftKey) redoOnce(); else undoOnce();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") { e.preventDefault(); redoOnce(); return; }
  if (e.key === "Escape") { mergeFrom = null; drawBoxes(); syncSide(); return; }
  if (typing) return;
  if (e.key === "v") setTool("select");
  if (e.key === "n") setTool("draw");
  if (e.key === "m") mergeSelected();
  if (e.key === "Delete" || e.key === "Backspace") deleteSelected();
});

function pushDone(name) {
  if (!name) return;
  let row = [...$("queue").children].find((x) => x.dataset.name === name);
  if (!row) {
    row = document.createElement("div"); row.className = "qrow"; row.dataset.name = name;
    row.textContent = "✓ " + name; $("queue").prepend(row);
    while ($("queue").children.length > 8) $("queue").lastChild.remove();
  }
}

async function translateLocked() {
  if (!sessionId) { alert("请先 OCR 并确认框"); return; }
  saveCfg(true); await savePage(true);
  $("translatebtn").disabled = true; $("download").style.display = "none";
  $("queue").innerHTML = ""; $("batchPreview").style.display = "none";
  $("barfill").style.width = "0%"; $("statTotal").textContent = "0"; $("statDone").textContent = "0"; $("statFail").textContent = "0";
  $("status").textContent = "正在按锁定框创建翻译任务…";
  try {
    const fd = new FormData();
    for (const k of ["endpoint", "model", "key", "batch", "maxout", "timeout"]) fd.append(k, $(k).value);
    const r = await fetch("/api/review_translate/" + sessionId, { method: "POST", body: fd });
    const d = await r.json();
    if (!r.ok || !d.ok) throw Error(d.error || "创建任务失败");
    currentTask = d.task_id; lastCompleted = 0; pollBatch();
  } catch (e) {
    $("status").textContent = "失败：" + e.message; $("translatebtn").disabled = false;
  }
}

async function pollBatch() {
  if (!currentTask) return;
  try {
    const r = await fetch("/api/batch_status/" + currentTask + "?t=" + Date.now(), { cache: "no-store" });
    const d = await r.json();
    if (!r.ok || !d.ok) throw Error(d.error || "读取进度失败");
    const total = d.total || 0, done = d.completed || 0, pct = total ? Math.min(100, Math.round(done * 100 / total)) : 0;
    $("barfill").style.width = pct + "%";
    $("statTotal").textContent = total; $("statDone").textContent = done; $("statFail").textContent = d.failed || 0;
    $("status").textContent = d.message || "";
    $("currentStatus").textContent = "状态：" + (d.current_status || d.status || "") + "\n进度：" + done + "/" + total;
    $("currentName").textContent = d.current_name ? ("第 " + (d.current || done) + " / " + total + "：" + d.current_name) : "尚未开始";
    if (done > lastCompleted) {
      lastCompleted = done;
      if (d.current_name) pushDone(d.current_name);
      $("batchPreview").src = (d.preview_url || ("/api/task_preview/" + currentTask)) + "?t=" + Date.now();
      $("batchPreview").style.display = "block";
    }
    if (d.status === "done") {
      $("download").href = d.download; $("download").style.display = "inline-block";
      $("download").textContent = "下载完成的 ZIP / CBZ";
      $("translatebtn").disabled = false; currentTask = null; return;
    }
    if (d.status === "error") throw Error(d.error || d.message || "批量任务失败");
    setTimeout(pollBatch, 800);
  } catch (e) {
    $("status").textContent = "失败：" + e.message;
    $("currentStatus").textContent = "任务停止：" + e.message;
    $("translatebtn").disabled = false; currentTask = null;
  }
}
