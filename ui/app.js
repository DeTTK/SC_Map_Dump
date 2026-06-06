const CHUNK_BLOCKS = 16;
const REGION_BLOCKS = 512;

const els = {
  status: document.getElementById("status"),
  form: document.getElementById("config-form"),
  scanBtn: document.getElementById("scan-btn"),
  paths: document.getElementById("paths"),
  worlds: document.getElementById("worlds"),
  includeAllWorlds: document.getElementById("include-all-worlds"),
  excludeAllWorlds: document.getElementById("exclude-all-worlds"),
  scanStats: document.getElementById("scan-stats"),
  selectionStats: document.getElementById("selection-stats"),
  selectAll: document.getElementById("select-all"),
  selectView: document.getElementById("select-view"),
  clearSelection: document.getElementById("clear-selection"),
  exportForm: document.getElementById("export-form"),
  jobs: document.getElementById("jobs"),
  canvas: document.getElementById("map"),
  hud: document.getElementById("hud-main"),
};

const ctx = els.canvas.getContext("2d");
const tileImages = new Map();
const EXCLUDED_WORLDS_STORAGE = "map_dump_mdat_excluded_worlds";

const state = {
  config: {},
  maps: [],
  scan: null,
  worldMeta: new Map(),
  chunks: [],
  chunkSet: new Set(),
  selected: new Set(),
  excludedWorlds: new Set(),
  view: { centerX: 0, centerZ: 0, scale: 0.35 },
  drag: null,
  selectionBox: null,
};

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

function key(x, z) {
  return `${x},${z}`;
}

function parseKey(value) {
  return value.split(",").map(Number);
}

function pointerCanvasPos(e) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * devicePixelRatio,
    y: (e.clientY - rect.top) * devicePixelRatio,
  };
}

function worldToScreen(x, z) {
  return {
    x: (x - state.view.centerX) * state.view.scale + els.canvas.width / 2,
    y: (z - state.view.centerZ) * state.view.scale + els.canvas.height / 2,
  };
}

function screenToWorld(sx, sy) {
  return {
    x: (sx - els.canvas.width / 2) / state.view.scale + state.view.centerX,
    z: (sy - els.canvas.height / 2) / state.view.scale + state.view.centerZ,
  };
}

function resize() {
  els.canvas.width = Math.max(1, Math.floor(els.canvas.clientWidth * devicePixelRatio));
  els.canvas.height = Math.max(1, Math.floor(els.canvas.clientHeight * devicePixelRatio));
  draw();
}

function fitScan() {
  if (!state.scan || !state.chunks.length) return;
  const b = state.scan.active_bounds || state.scan.bounds;
  const x0 = b.min_x * CHUNK_BLOCKS;
  const z0 = b.min_z * CHUNK_BLOCKS;
  const x1 = (b.max_x + 1) * CHUNK_BLOCKS;
  const z1 = (b.max_z + 1) * CHUNK_BLOCKS;
  state.view.centerX = (x0 + x1) / 2;
  state.view.centerZ = (z0 + z1) / 2;
  const sx = els.canvas.width / Math.max(1, x1 - x0);
  const sz = els.canvas.height / Math.max(1, z1 - z0);
  state.view.scale = Math.max(0.03, Math.min(1.8, Math.min(sx, sz) * 0.88));
}

function getTile(olmap, rx, rz) {
  if (!olmap) return null;
  const tileKey = `${olmap}/${rx}/${rz}`;
  if (tileImages.has(tileKey)) return tileImages.get(tileKey);
  const img = new Image();
  img.src = `/maps/${encodeURIComponent(olmap)}/tile/${rx}/${rz}.png`;
  img.onload = () => draw();
  img.onerror = () => tileImages.set(tileKey, null);
  tileImages.set(tileKey, img);
  return img;
}

function drawTiles() {
  const olmap = state.config.olmap || "";
  if (!olmap) return;
  const tl = screenToWorld(0, 0);
  const br = screenToWorld(els.canvas.width, els.canvas.height);
  const minRx = Math.floor(Math.min(tl.x, br.x) / REGION_BLOCKS) - 1;
  const maxRx = Math.floor(Math.max(tl.x, br.x) / REGION_BLOCKS) + 1;
  const minRz = Math.floor(Math.min(tl.z, br.z) / REGION_BLOCKS) - 1;
  const maxRz = Math.floor(Math.max(tl.z, br.z) / REGION_BLOCKS) + 1;
  const size = REGION_BLOCKS * state.view.scale;
  for (let rx = minRx; rx <= maxRx; rx++) {
    for (let rz = minRz; rz <= maxRz; rz++) {
      const img = getTile(olmap, rx, rz);
      const pos = worldToScreen(rx * REGION_BLOCKS, rz * REGION_BLOCKS);
      if (img && img.complete && img.naturalWidth) {
        ctx.drawImage(img, pos.x, pos.y, size, size);
      }
    }
  }
}

function drawChunks() {
  const size = Math.max(1, CHUNK_BLOCKS * state.view.scale);
  for (const chunk of state.chunks) {
    const pos = worldToScreen(chunk.x * CHUNK_BLOCKS, chunk.z * CHUNK_BLOCKS);
    if (pos.x < -size || pos.y < -size || pos.x > els.canvas.width || pos.y > els.canvas.height) continue;
    const selected = state.selected.has(key(chunk.x, chunk.z));
    ctx.fillStyle = selected ? "rgba(241,182,87,.50)" : "rgba(65,217,167,.30)";
    ctx.fillRect(pos.x, pos.y, size, size);
    if (selected || state.view.scale > 0.45) {
      ctx.strokeStyle = selected ? "rgba(241,182,87,.95)" : "rgba(65,217,167,.45)";
      ctx.lineWidth = 1;
      ctx.strokeRect(pos.x + 0.5, pos.y + 0.5, Math.max(1, size - 1), Math.max(1, size - 1));
    }
  }
}

function drawSelectionBox() {
  if (!state.selectionBox) return;
  const { sx, sy, ex, ey, mode } = state.selectionBox;
  ctx.strokeStyle = mode === "remove" ? "rgba(255,109,109,.95)" : "rgba(241,182,87,.95)";
  ctx.lineWidth = 1.5 * devicePixelRatio;
  ctx.setLineDash([5 * devicePixelRatio, 4 * devicePixelRatio]);
  ctx.strokeRect(Math.min(sx, ex), Math.min(sy, ey), Math.abs(ex - sx), Math.abs(ey - sy));
  ctx.setLineDash([]);
}

function draw() {
  if (!els.canvas.width) return;
  ctx.fillStyle = "#050708";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  drawTiles();
  drawChunks();
  drawSelectionBox();
  updateStats();
}

function chunksInRect(sx, sy, ex, ey) {
  const a = screenToWorld(Math.min(sx, ex), Math.min(sy, ey));
  const b = screenToWorld(Math.max(sx, ex), Math.max(sy, ey));
  const cx0 = Math.floor(a.x / CHUNK_BLOCKS);
  const cz0 = Math.floor(a.z / CHUNK_BLOCKS);
  const cx1 = Math.floor(b.x / CHUNK_BLOCKS);
  const cz1 = Math.floor(b.z / CHUNK_BLOCKS);
  const out = [];
  for (let cx = cx0; cx <= cx1; cx++) {
    for (let cz = cz0; cz <= cz1; cz++) {
      const k = key(cx, cz);
      if (state.chunkSet.has(k)) out.push(k);
    }
  }
  return out;
}

function visibleChunks() {
  const a = screenToWorld(0, 0);
  const b = screenToWorld(els.canvas.width, els.canvas.height);
  return chunksInRect(
    worldToScreen(Math.min(a.x, b.x), Math.min(a.z, b.z)).x,
    worldToScreen(Math.min(a.x, b.x), Math.min(a.z, b.z)).y,
    worldToScreen(Math.max(a.x, b.x), Math.max(a.z, b.z)).x,
    worldToScreen(Math.max(a.x, b.x), Math.max(a.z, b.z)).y,
  );
}

function setScan(scan, { fit = true } = {}) {
  state.scan = scan;
  const localExcluded = loadLocalExcludedWorlds();
  if (localExcluded) state.excludedWorlds = localExcluded;
  else if (scan.excluded_worlds) state.excludedWorlds = new Set(scan.excluded_worlds);
  state.worldMeta = new Map((scan.worlds || []).map(w => [w.name, w]));
  rebuildActiveChunks();
  renderWorlds(scan.worlds || []);
  if (fit) fitScan();
  draw();
}

function rebuildActiveChunks() {
  if (!state.scan) return;
  const byCoord = new Map();
  let raw = 0;
  for (const chunk of state.scan.source_chunks || []) {
    if (state.excludedWorlds.has(chunk.world)) continue;
    raw++;
    const k = key(chunk.x, chunk.z);
    const current = byCoord.get(k);
    if (!current || compareChunkSource(chunk, current) > 0) {
      byCoord.set(k, chunk);
    }
  }
  state.chunks = [...byCoord.values()].sort((a, b) => a.x - b.x || a.z - b.z);
  state.chunkSet = new Set(state.chunks.map(c => key(c.x, c.z)));
  state.selected = new Set([...state.selected].filter(k => state.chunkSet.has(k)));
  let minX = 0, maxX = 0, minZ = 0, maxZ = 0;
  if (state.chunks.length) {
    minX = maxX = state.chunks[0].x;
    minZ = maxZ = state.chunks[0].z;
    for (const chunk of state.chunks) {
      if (chunk.x < minX) minX = chunk.x;
      if (chunk.x > maxX) maxX = chunk.x;
      if (chunk.z < minZ) minZ = chunk.z;
      if (chunk.z > maxZ) maxZ = chunk.z;
    }
  }
  state.scan.active_chunk_count = state.chunks.length;
  state.scan.active_raw_chunk_count = raw;
  state.scan.active_bounds = {
    min_x: minX,
    max_x: maxX,
    min_z: minZ,
    max_z: maxZ,
  };
}

function compareChunkSource(a, b) {
  const aw = state.worldMeta.get(a.world) || {};
  const bw = state.worldMeta.get(b.world) || {};
  const am = Number(aw.mtime || 0);
  const bm = Number(bw.mtime || 0);
  if (am !== bm) return am - bm;
  return String(a.world).localeCompare(String(b.world));
}

function updateStats() {
  if (!state.scan) {
    els.scanStats.textContent = "no scan yet";
    els.selectionStats.textContent = "0 selected";
    els.hud.textContent = "no chunks";
    return;
  }
  const b = state.scan.active_bounds || state.scan.bounds;
  const regions = (state.scan.worlds || []).filter(w => !state.excludedWorlds.has(w.name)).reduce((sum, w) => sum + Number(w.regions || 0), 0);
  els.scanStats.textContent = `${state.chunks.length} chunks (${state.scan.active_raw_chunk_count || 0} raw), ${regions} regions, x ${b.min_x}..${b.max_x}, z ${b.min_z}..${b.max_z}`;
  els.selectionStats.textContent = `${state.selected.size} selected`;
  els.hud.textContent = `${state.chunkSet.size} chunks | ${state.selected.size} selected | scale ${state.view.scale.toFixed(2)}`;
}

function renderMaps() {
  const select = els.form.elements.olmap;
  const current = state.config.olmap || "";
  select.innerHTML = '<option value="">map overlay</option>';
  for (const map of state.maps) {
    const opt = document.createElement("option");
    opt.value = map.name;
    opt.textContent = `${map.name} (${map.regions})`;
    if (map.name === current) opt.selected = true;
    select.appendChild(opt);
  }
}

function renderPaths(data) {
  const lines = [];
  if (data.map_cache) lines.push(`map_cache: ${data.map_cache}`);
  if (state.config.olmap) lines.push(`2d map: ${state.config.olmap}`);
  if (data.textures) {
    for (const [name, info] of Object.entries(data.textures)) {
      lines.push(`${info.exists ? "ok" : "missing"} ${name}: ${info.path}`);
    }
  }
  if (data.error) lines.push(`error: ${data.error}`);
  els.paths.textContent = lines.join("\n");
}

function renderWorlds(worlds) {
  els.worlds.innerHTML = "";
  for (const world of worlds) {
    const li = document.createElement("li");
    li.className = "world-row";
    const checked = !state.excludedWorlds.has(world.name);
    li.innerHTML = `
      <label>
        <input type="checkbox" ${checked ? "checked" : ""} />
        <span>${escapeHtml(world.name)}</span>
      </label>
      <span class="meta">${world.chunks || 0} chunks / ${world.regions} reg</span>
    `;
    li.querySelector("input").addEventListener("change", async ev => {
      if (ev.currentTarget.checked) state.excludedWorlds.delete(world.name);
      else state.excludedWorlds.add(world.name);
      persistExcludedWorlds();
      rebuildActiveChunks();
      draw();
    });
    els.worlds.appendChild(li);
  }
}

async function loadConfig() {
  const data = await api("/config");
  state.config = data.config || {};
  state.excludedWorlds = new Set(state.config.excluded_worlds || []);
  const localExcluded = loadLocalExcludedWorlds();
  if (localExcluded) state.excludedWorlds = localExcluded;
  els.form.elements.game_dir.value = state.config.game_dir || "";
  els.form.elements.blender.value = state.config.blender || "";
  await loadMaps();
  renderPaths(data);
  if (data.worlds) renderWorlds(data.worlds);
}

async function loadMaps() {
  state.maps = await api("/maps");
  renderMaps();
}

async function saveConfig() {
  const body = {
    game_dir: els.form.elements.game_dir.value,
    blender: els.form.elements.blender.value,
    olmap: els.form.elements.olmap.value,
    excluded_worlds: [...state.excludedWorlds],
  };
  const data = await api("/config", { method: "POST", body: JSON.stringify(body) });
  state.config = data.config || body;
  persistExcludedWorlds();
  renderMaps();
}

function loadLocalExcludedWorlds() {
  try {
    const raw = localStorage.getItem(EXCLUDED_WORLDS_STORAGE);
    if (!raw) return null;
    const values = JSON.parse(raw);
    if (!Array.isArray(values)) return null;
    return new Set(values.map(String));
  } catch {
    return null;
  }
}

function persistExcludedWorlds() {
  localStorage.setItem(EXCLUDED_WORLDS_STORAGE, JSON.stringify([...state.excludedWorlds]));
}

async function scan(force = false, opts = {}) {
  els.status.textContent = "scanning...";
  const data = force ? await api("/scan", { method: "POST", body: "{}" }) : await api("/scan");
  setScan(data, opts);
  renderPaths(data);
  els.status.textContent = "scan ready";
}

async function refreshJobs() {
  const data = await api("/jobs");
  els.jobs.innerHTML = "";
  for (const job of data.jobs || []) {
    const li = document.createElement("li");
    const status = job.running ? "running" : job.error ? "failed" : "done";
    li.innerHTML = `
      <div><b>${escapeHtml(job.name)}</b> <span class="${job.error ? "bad" : job.running ? "warn" : "ok"}">${status}</span></div>
      <div class="meta">${job.chunks} chunks${job.blend ? `\n${escapeHtml(job.blend)}` : ""}${job.error ? `\n${escapeHtml(job.error)}` : ""}</div>
    `;
    els.jobs.appendChild(li);
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}

els.form.addEventListener("submit", async ev => {
  ev.preventDefault();
  try {
    els.status.textContent = "saving...";
    await saveConfig();
    await scan(true);
  } catch (err) {
    els.status.textContent = err.message;
  }
});

els.form.elements.olmap.addEventListener("change", async () => {
  try {
    await saveConfig();
    draw();
  } catch (err) {
    els.status.textContent = err.message;
  }
});

els.scanBtn.addEventListener("click", async () => {
  try {
    await saveConfig();
    await scan(true);
  } catch (err) {
    els.status.textContent = err.message;
  }
});
els.includeAllWorlds.addEventListener("click", async () => {
  state.excludedWorlds.clear();
  persistExcludedWorlds();
  renderWorlds(state.scan?.worlds || []);
  rebuildActiveChunks();
  draw();
});
els.excludeAllWorlds.addEventListener("click", async () => {
  state.excludedWorlds = new Set((state.scan?.worlds || []).map(w => w.name));
  persistExcludedWorlds();
  renderWorlds(state.scan?.worlds || []);
  rebuildActiveChunks();
  draw();
});
els.selectAll.addEventListener("click", () => {
  state.selected = new Set(state.chunkSet);
  draw();
});
els.selectView.addEventListener("click", () => {
  for (const k of visibleChunks()) state.selected.add(k);
  draw();
});
els.clearSelection.addEventListener("click", () => {
  state.selected.clear();
  draw();
});

els.exportForm.addEventListener("submit", async ev => {
  ev.preventDefault();
  if (!state.selected.size) {
    els.status.textContent = "select chunks first";
    return;
  }
  const options = { name: els.exportForm.elements.name.value };
  const chunks = [...state.selected].map(parseKey);
  els.status.textContent = "export queued";
  await saveConfig();
  await api("/export", { method: "POST", body: JSON.stringify({ chunks, options }) });
  await refreshJobs();
});

els.canvas.addEventListener("mousedown", ev => {
  const pos = pointerCanvasPos(ev);
  if (ev.button === 1 || ev.button === 2 || ev.shiftKey) {
    state.drag = { mode: "pan", x: ev.clientX, y: ev.clientY };
    return;
  }
  state.drag = { mode: "select" };
  state.selectionBox = { sx: pos.x, sy: pos.y, ex: pos.x, ey: pos.y, mode: ev.altKey ? "remove" : "add" };
});
els.canvas.addEventListener("contextmenu", ev => ev.preventDefault());
window.addEventListener("mousemove", ev => {
  if (!state.drag) return;
  if (state.drag.mode === "pan") {
    const dx = (ev.clientX - state.drag.x) * devicePixelRatio;
    const dy = (ev.clientY - state.drag.y) * devicePixelRatio;
    state.view.centerX -= dx / state.view.scale;
    state.view.centerZ -= dy / state.view.scale;
    state.drag.x = ev.clientX;
    state.drag.y = ev.clientY;
    draw();
    return;
  }
  if (state.selectionBox) {
    const pos = pointerCanvasPos(ev);
    state.selectionBox.ex = pos.x;
    state.selectionBox.ey = pos.y;
    draw();
  }
});
window.addEventListener("mouseup", () => {
  if (state.selectionBox) {
    const box = state.selectionBox;
    const cells = chunksInRect(box.sx, box.sy, box.ex, box.ey);
    for (const k of cells) {
      if (box.mode === "remove") state.selected.delete(k);
      else state.selected.add(k);
    }
  }
  state.drag = null;
  state.selectionBox = null;
  draw();
});
els.canvas.addEventListener("wheel", ev => {
  ev.preventDefault();
  const pos = pointerCanvasPos(ev);
  const before = screenToWorld(pos.x, pos.y);
  state.view.scale *= ev.deltaY < 0 ? 1.15 : 0.87;
  state.view.scale = Math.max(0.02, Math.min(8, state.view.scale));
  const after = screenToWorld(pos.x, pos.y);
  state.view.centerX += before.x - after.x;
  state.view.centerZ += before.z - after.z;
  draw();
}, { passive: false });

window.addEventListener("resize", resize);
setInterval(refreshJobs, 2000);

try {
  await loadConfig();
  await scan(false);
} catch (err) {
  els.status.textContent = err.message;
}
resize();
refreshJobs().catch(() => {});
