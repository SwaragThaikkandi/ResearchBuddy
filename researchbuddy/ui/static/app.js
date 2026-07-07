/* ResearchBuddy UI — dependency-free vanilla JS (offline, no CDN). */
"use strict";

const $ = (sel) => document.querySelector(sel);
const api = async (path, body) => {
  const opts = body === undefined ? {} :
    { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) };
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status + "";
    try { msg = (await r.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
};

/* ── Global progress overlay ──────────────────────────────────────────── */
let progTimer = null, progOps = 0;

function progressRender(p) {
  const box = $("#progress"), fill = $("#progress-fill");
  if (!p || !p.active) {
    if (progOps === 0) box.classList.add("hidden");
    return;
  }
  box.classList.remove("hidden");
  $("#progress-text").textContent = p.text || "working…";
  if (p.pct == null) {
    fill.classList.add("indeterminate");
  } else {
    fill.classList.remove("indeterminate");
    fill.style.width = Math.max(3, Math.min(100, p.pct * 100)) + "%";
  }
}
function progressStart(initialText) {
  progOps++;
  progressRender({ active: true, pct: null, text: initialText });
  if (!progTimer)
    progTimer = setInterval(async () => {
      try { progressRender(await api("/api/progress")); } catch (e) {}
    }, 700);
}
function progressEnd() {
  progOps = Math.max(0, progOps - 1);
  if (progOps === 0) {
    clearInterval(progTimer); progTimer = null;
    $("#progress").classList.add("hidden");
  }
}
/* Wrap a long-running op: shows the bar, polls server-side progress text,
   always hides on completion/error. */
async function withProgress(initialText, fn) {
  progressStart(initialText);
  try { return await fn(); }
  finally { progressEnd(); }
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
function showTab(name) {
  document.querySelectorAll("#tabs button").forEach(b =>
    b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach(t =>
    t.classList.toggle("active", t.id === "tab-" + name));
  if (name === "graph") loadGraph();
  if (name === "watches") { loadWatches(); loadSentinel(); }
  if (name === "collab") loadCollab();
  if (name === "services") loadServices();
  if (name === "evolution") loadEvolution();
}
document.querySelectorAll("#tabs button").forEach(btn => {
  btn.onclick = () => showTab(btn.dataset.tab);
});

/* ── Stats bar ────────────────────────────────────────────────────────── */
async function loadStats() {
  try {
    const s = await api("/api/stats");
    $("#statbar").textContent =
      `${s.total_papers} papers · ${s.rated_papers} rated · ` +
      `${s.niche_clusters ?? 0} niches · sem ${s.semantic_edges ?? 0} / ` +
      `cit ${s.citation_edges ?? 0} edges`;
  } catch (e) { $("#statbar").textContent = "stats unavailable"; }
}

/* ── Result cards + rating ────────────────────────────────────────────── */
function resultCard(p) {
  const div = document.createElement("div");
  div.className = "card";
  const tags = [];
  if (p.label === "explore") tags.push('<span class="res-tag explore">EXPLORE</span>');
  if (p.score != null) tags.push(`<span class="res-tag">match ${(p.score * 100).toFixed(0)}%</span>`);
  if (p.peer_reviewed === true) tags.push('<span class="res-tag">peer-reviewed</span>');
  if (p.peer_reviewed === false) tags.push('<span class="res-tag">preprint</span>');
  if (p.cited_by != null) tags.push(`<span class="res-tag">${p.cited_by} cites</span>`);
  const link = p.doi ? `https://doi.org/${p.doi}` : p.url;
  div.innerHTML = `
    <div class="res-title">${esc(p.title)}</div>
    <div class="res-meta">${esc((p.authors || []).join(", "))} (${p.year ?? "?"})
      ${link ? `· <a href="${esc(link)}" target="_blank">open</a>` : ""}</div>
    <div>${tags.join("")}</div>
    <div class="res-abs">${esc(p.abstract || "")}</div>
    <div class="rate-row"><span class="res-meta">rate:</span></div>`;
  const row = div.querySelector(".rate-row");
  let rated = false;
  for (let i = 1; i <= 10; i++) {
    const b = document.createElement("button");
    b.textContent = i;
    b.onclick = async () => {
      try {
        await api("/api/rate", { token: p.token, rating: i });
        row.querySelectorAll("button").forEach(x => x.classList.remove("rated"));
        b.classList.add("rated");
        rated = true;
        pdfBtn.classList.remove("hidden");
        loadStats();
      } catch (e) { alert("rate failed: " + e.message); }
    };
    row.appendChild(b);
  }
  // Optional PDF attach (CLI parity): after rating, offer to ingest the PDF
  // so GROBID adds section embeddings + parsed references to the node.
  const pdfInput = document.createElement("input");
  pdfInput.type = "file"; pdfInput.accept = ".pdf,application/pdf";
  pdfInput.classList.add("hidden");
  const pdfBtn = document.createElement("button");
  pdfBtn.textContent = "📎 attach PDF";
  pdfBtn.title = "optional — GROBID parses it into section embeddings + " +
    "references, making this a full graph node";
  pdfBtn.classList.add("hidden");        // appears once rated
  pdfBtn.onclick = () => pdfInput.click();
  pdfInput.onchange = () => {
    const f = pdfInput.files[0];
    if (!f) return;
    withProgress(`Parsing ${f.name} with GROBID…`, async () => {
      const fd = new FormData();
      fd.append("token", p.token);
      fd.append("file", f);
      const r = await fetch("/api/attach_pdf", { method: "POST", body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.status);
      const info = await r.json();
      pdfBtn.textContent =
        `✓ ${info.n_sections} sections · ${info.n_refs} refs [${info.parser}]`;
      pdfBtn.disabled = true;
      loadStats();
    }).catch(e => alert("PDF ingest failed: " + e.message))
      .finally(() => { pdfInput.value = ""; });
  };
  row.appendChild(pdfBtn);
  row.appendChild(pdfInput);
  return div;
}
function renderResults(el, results) {
  el.innerHTML = "";
  if (!results.length) { el.innerHTML = '<p class="note">No results.</p>'; return; }
  results.forEach(p => el.appendChild(resultCard(p)));
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/* ── Graph tab: force-directed canvas ─────────────────────────────────── */
let G = { nodes: [], links: [] }, sim = null, selected = null;

async function loadGraph() {
  const minw = $("#g-minw").value;
  const data = await api(`/api/graph?min_weight=${minw}`);
  G = data;
  const idx = {}; G.nodes.forEach((n, i) => { idx[n.id] = i;
    n.x = Math.random() * 800; n.y = Math.random() * 600; n.vx = 0; n.vy = 0; });
  G.links = G.links.filter(l => idx[l.s] != null && idx[l.t] != null)
    .map(l => ({ ...l, si: idx[l.s], ti: idx[l.t] }));
  startSim();
}

function startSim() {
  const canvas = $("#graph-canvas");
  const ctx = canvas.getContext("2d");
  const wrap = $("#graph-wrap");
  canvas.width = wrap.clientWidth; canvas.height = wrap.clientHeight;
  let W = canvas.width, H = canvas.height;
  let ticks = 0;
  const N = G.nodes.length;
  if (sim) cancelAnimationFrame(sim);

  function step() {
    // spring-electric layout: repulsion (grid-free O(N^2), fine to ~400)
    const k = Math.sqrt(W * H / Math.max(N, 1)) * 0.6;
    for (let i = 0; i < N; i++) {
      const a = G.nodes[i];
      for (let j = i + 1; j < N; j++) {
        const b = G.nodes[j];
        let dx = a.x - b.x, dy = a.y - b.y;
        let d2 = dx * dx + dy * dy + 0.01, d = Math.sqrt(d2);
        const f = (k * k) / d2 * 6;
        dx = dx / d * f; dy = dy / d * f;
        a.vx += dx; a.vy += dy; b.vx -= dx; b.vy -= dy;
      }
    }
    for (const l of G.links) {           // attraction along edges
      const a = G.nodes[l.si], b = G.nodes[l.ti];
      const dx = b.x - a.x, dy = b.y - a.y;
      const d = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const f = (d - k) / d * 0.02 * (l.w || 1);
      a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    for (const n of G.nodes) {           // integrate + center gravity + damp
      n.vx += (W / 2 - n.x) * 0.001; n.vy += (H / 2 - n.y) * 0.001;
      n.x += Math.max(-8, Math.min(8, n.vx)); n.vx *= 0.6;
      n.y += Math.max(-8, Math.min(8, n.vy)); n.vy *= 0.6;
      n.x = Math.max(8, Math.min(W - 8, n.x));
      n.y = Math.max(8, Math.min(H - 8, n.y));
    }
    draw();
    if (++ticks < 300) sim = requestAnimationFrame(step);
  }

  function color(n) {
    if (n.kind && n.kind !== "paper") return "#c084fc";
    if (n.rating != null) return "#43d17c";
    if (n.fulltext) return "#ffb454";
    return "#4da3ff";
  }
  function radius(n) { return 4 + Math.min(8, (n.deg || 0) * 0.7); }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    ctx.lineWidth = 1;
    for (const l of G.links) {
      const a = G.nodes[l.si], b = G.nodes[l.ti];
      ctx.strokeStyle = l.layer === "citation"
        ? "rgba(255,180,84,0.25)" : "rgba(77,163,255,0.18)";
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
    }
    for (const n of G.nodes) {
      ctx.fillStyle = color(n);
      ctx.beginPath(); ctx.arc(n.x, n.y, radius(n), 0, 7); ctx.fill();
      if (n === selected) {
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(n.x, n.y, radius(n) + 3, 0, 7); ctx.stroke();
      }
    }
  }

  canvas.onclick = async (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    selected = G.nodes.find(n =>
      (n.x - x) ** 2 + (n.y - y) ** 2 < (radius(n) + 4) ** 2) || null;
    draw();
    const panel = $("#node-panel");
    if (!selected) { panel.classList.add("hidden"); return; }
    const d = await api(`/api/paper/${encodeURIComponent(selected.id)}`);
    panel.classList.remove("hidden");
    panel.innerHTML = `<div class="res-title">${esc(d.title)}</div>
      <div class="res-meta">${esc((d.authors || []).join(", "))} (${d.year ?? "?"})</div>
      <div class="res-meta">${d.rating != null ? "rated " + d.rating + "/10 · " : ""}
        ${d.has_fulltext ? "full text · " : ""}${esc(d.venue || "")}</div>
      <div class="res-abs">${esc(d.abstract || "")}</div>
      ${d.doi ? `<a href="https://doi.org/${esc(d.doi)}" target="_blank">doi</a>` : ""}`;
  };
  step();
}
$("#g-reload").onclick = loadGraph;
$("#g-minw").onchange = loadGraph;
$("#g-rebuild").onclick = async () => {
  $("#g-rebuild").disabled = true;
  try { await api("/api/rebuild", {}); await loadGraph(); await loadStats(); }
  finally { $("#g-rebuild").disabled = false; }
};

/* ── Library tab: PDF upload + folder import ──────────────────────────── */
let queuedFiles = [];
const drop = $("#l-drop"), fileInput = $("#l-files");

function drawQueue() {
  const q = $("#l-queue"); q.innerHTML = "";
  queuedFiles.forEach((f, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = `${esc(f.name)} <span class="res-meta">` +
      `${(f.size / 1048576).toFixed(1)} MB</span><b>×</b>`;
    c.querySelector("b").onclick = () => {
      queuedFiles.splice(i, 1); drawQueue(); };
    q.appendChild(c);
  });
  $("#l-upload").disabled = !queuedFiles.length;
}
function queuePdfs(list) {
  for (const f of list)
    if (/\.pdf$/i.test(f.name)) queuedFiles.push(f);
  drawQueue();
}
drop.onclick = () => fileInput.click();
fileInput.onchange = () => { queuePdfs(fileInput.files); fileInput.value = ""; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("drag"); };
drop.ondragleave = () => drop.classList.remove("drag");
drop.ondrop = (e) => {
  e.preventDefault(); drop.classList.remove("drag");
  queuePdfs(e.dataTransfer.files);
};
$("#l-upload").onclick = async () => {
  const fd = new FormData();
  queuedFiles.forEach(f => fd.append("files", f));
  fd.append("kind", $("#l-kind").value);
  $("#l-status").textContent = "";
  $("#l-upload").disabled = true;
  try {
    const body = await withProgress(
      `Uploading ${queuedFiles.length} PDF(s)…`, async () => {
        const r = await fetch("/api/upload_pdfs", { method: "POST", body: fd });
        if (!r.ok) throw new Error((await r.json()).detail || r.status);
        return r.json();
      });
    $("#l-status").innerHTML =
      `<span class="ok">${body.added} added</span> of ${body.uploaded} ` +
      `uploaded ${body.note ? "· " + esc(body.note) : ""}`;
    queuedFiles = []; drawQueue();
    loadStats(); loadGraph();
  } catch (e) {
    $("#l-status").innerHTML = `<span class="err">error: ${esc(e.message)}</span>`;
    $("#l-upload").disabled = false;
  }
};
$("#l-folder-run").onclick = async () => {
  const path = $("#l-folder").value.trim();
  if (!path) return;
  $("#l-folder-status").textContent = "importing folder…";
  try {
    const r = await api("/api/import_folder", { path });
    $("#l-folder-status").innerHTML =
      `<span class="ok">${r.added} added</span>`;
    loadStats(); loadGraph();
  } catch (e) {
    $("#l-folder-status").innerHTML =
      `<span class="err">error: ${esc(e.message)}</span>`;
  }
};

/* ── Discover tab ─────────────────────────────────────────────────────── */
const focusIds = new Map();     // token -> title
$("#d-focus-q").oninput = async (ev) => {
  const q = ev.target.value.trim();
  const box = $("#d-focus-hits");
  if (q.length < 2) { box.innerHTML = ""; return; }
  const hits = await api(`/api/library_search?q=${encodeURIComponent(q)}`);
  box.innerHTML = "";
  hits.forEach(h => {
    const d = document.createElement("div");
    d.textContent = `${h.title} (${h.year ?? "?"})`;
    d.onclick = () => { focusIds.set(h.token, h.title); drawChips();
      box.innerHTML = ""; $("#d-focus-q").value = ""; };
    box.appendChild(d);
  });
};
function drawChips() {
  const el = $("#d-focus-chips"); el.innerHTML = "";
  focusIds.forEach((title, id) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = esc(title.slice(0, 40)) + "<b>×</b>";
    c.querySelector("b").onclick = () => { focusIds.delete(id); drawChips(); };
    el.appendChild(c);
  });
}
$("#d-run").onclick = async () => {
  $("#d-status").textContent = "";
  $("#d-run").disabled = true;
  try {
    const r = await withProgress("Starting search…", () =>
      api("/api/search", {
        intent: $("#d-intent").value, keywords: $("#d-keywords").value,
        focus_ids: [...focusIds.keys()], n: 10 }));
    $("#d-status").textContent = `${r.n_fetched} fetched`;
    renderResults($("#d-results"), r.results);
  } catch (e) { $("#d-status").textContent = "error: " + e.message; }
  finally { $("#d-run").disabled = false; }
};

/* ── Snowball tab ─────────────────────────────────────────────────────── */
$("#s-run").onclick = async () => {
  const dirs = [];
  if ($("#s-back").checked) dirs.push("backward");
  if ($("#s-fwd").checked) dirs.push("forward");
  $("#s-status").textContent = "";
  $("#s-run").disabled = true;
  try {
    const n = Math.max(1, Math.min(30, +$("#s-n").value || 10));
    const r = await withProgress("Starting snowball round…", () =>
      api("/api/snowball", { directions: dirs, n,
        reset_frontier: $("#s-reset").checked }));
    const st = r.stats, b = $("#s-stats");
    b.classList.remove("hidden");
    b.innerHTML = st.error ? `<span class="warn">${esc(st.error)}</span>` :
      `fetched ${st.fetched} → <b>${st.new_unique} new</b> · saturation ` +
      `${(st.saturation_ratio * 100).toFixed(0)}% · seeds remaining ` +
      `${st.seeds_remaining ?? "?"} ` +
      (st.saturated ? '<span class="ok">SATURATED — coverage solid</span>' : "");
    renderResults($("#s-results"), r.results);
    $("#s-status").textContent = "";
  } catch (e) { $("#s-status").textContent = "error: " + e.message; }
  finally { $("#s-run").disabled = false; }
};

/* ── Harvest tab ──────────────────────────────────────────────────────── */
$("#h-run").onclick = async () => {
  $("#h-status").textContent = "";
  $("#h-run").disabled = true;
  try {
    const r = await withProgress("Resolving open-access copies…", () =>
      api("/api/harvest", { n: +$("#h-n").value }));
    $("#h-status").innerHTML =
      `<span class="ok">${r.ingested} ingested</span> / ${r.downloaded} ` +
      `downloaded / ${r.checked} checked · no-OA ${r.no_oa}`;
    const log = $("#h-log");
    log.classList.remove("hidden");
    log.textContent = r.log.join("\n") +
      (r.errors.length ? "\nERRORS:\n" + r.errors.join("\n") : "");
    loadStats();
  } catch (e) { $("#h-status").textContent = "error: " + e.message; }
  finally { $("#h-run").disabled = false; }
};

/* ── Review tab ───────────────────────────────────────────────────────── */
$("#r-run").onclick = async () => {
  $("#r-status").textContent = "";
  $("#r-run").disabled = true;
  try {
    const r = await withProgress("Building your review…", () =>
      api("/api/review_pack", { use_llm: $("#r-llm").checked }));
    const out = $("#r-out");
    out.classList.remove("hidden");
    out.innerHTML = `Files saved to <code>${esc(r.path)}</code><br>` +
      r.files.map(f => `<span class="res-tag">${esc(f)}</span>`).join(" ");
    // Show the review right here, not just file paths.
    if (r.scaffold) {
      $("#r-scaffold").classList.remove("hidden");
      $("#r-scaffold").innerHTML = mdLite(r.scaffold);
    }
    if (r.prisma) {
      $("#r-prisma").classList.remove("hidden");
      $("#r-prisma").innerHTML = mdLite(r.prisma);
    }
    // And the thought map alongside it.
    $("#r-map-run").click();
  } catch (e) { $("#r-status").textContent = "error: " + e.message; }
  finally { $("#r-run").disabled = false; }
};

/* ── Reasoning tab ────────────────────────────────────────────────────── */
$("#q-run").onclick = async () => {
  const q = $("#q-input").value.trim();
  if (!q) return;
  $("#q-run").disabled = true; $("#q-status").textContent = "";
  try {
    const r = await withProgress("Reasoning over your collection…", () =>
      api("/api/query", { query: q }));
    $("#q-out").classList.remove("hidden");
    $("#q-relevant").innerHTML = r.relevant.map(p =>
      `<div class="banner"><b>${esc(p.title)}</b>
       <span class="res-meta">(${p.year ?? "?"}) · relevance
       ${(p.score * 100).toFixed(0)}% · ${p.degree} connections
       ${p.role ? "· " + esc(p.role) : ""}
       ${p.rating != null ? "· rated " + p.rating + "/10" : ""}</span></div>`
    ).join("") || '<p class="note">nothing relevant found</p>';
    $("#q-themes").innerHTML = r.themes.map(t =>
      `<div class="banner">${esc(t.id)} — ${t.n_papers} papers · match
       ${(t.match * 100).toFixed(0)}% · ${esc(t.maturity)} · density
       ${(t.density * 100).toFixed(0)}%<br>
       <span class="res-meta">key paper: ${esc(t.central)}</span></div>`
    ).join("") || '<p class="note">no theme profiles yet (rebuild hierarchy)</p>';
    $("#q-lineages").innerHTML =
      r.lineages.map(l => `<div class="banner"><span class="res-meta">
        ${esc(l.type)}:</span> ${l.titles.map(esc).join(" → ")}</div>`).join("") +
      r.connections.map(c => `<div class="banner">${esc(c[0])} ↔ ${esc(c[1])}
        <span class="res-meta">(${esc(c[2])})</span></div>`).join("") ||
      '<p class="note">none found</p>';
    $("#q-bridges").innerHTML =
      (r.bridges.length ? "<b>Bridges:</b> " + r.bridges.map(esc).join(" · ") : "") +
      (r.frontier.length ? "<br><b>Frontier:</b> " +
        r.frontier.map(f => `${esc(f[0])} (${(f[1] * 100).toFixed(0)}%)`).join(" · ") : "") +
      (r.narrative ? `<br><span class="res-meta">${esc(r.narrative)}</span>` : "") +
      (r.gap_note ? `<br><span class="warn">${esc(r.gap_note)}</span>` : "") ||
      '<p class="note">none found</p>';
    const rate = $("#q-rate"); rate.innerHTML = "";
    for (let i = 1; i <= 10; i++) {
      const b = document.createElement("button");
      b.textContent = i;
      b.onclick = async () => {
        await api("/api/query_feedback", { rating: i });
        rate.querySelectorAll("button").forEach(x => x.classList.remove("rated"));
        b.classList.add("rated");
      };
      rate.appendChild(b);
    }
  } catch (e) { $("#q-status").textContent = "error: " + e.message; }
  finally { $("#q-run").disabled = false; }
};

/* ── Tiny markdown-lite renderer (headings, tables, bold, code) ───────── */
function mdLite(md) {
  const lines = md.split("\n"); let html = "", inTable = false;
  for (const ln of lines) {
    if (/^\|/.test(ln)) {
      if (/^\|[\s\-|]+\|$/.test(ln)) continue;            // separator row
      const cells = ln.split("|").slice(1, -1).map(c => inline(c.trim()));
      if (!inTable) { html += "<table>"; inTable = true; }
      html += "<tr><td>" + cells.join("</td><td>") + "</td></tr>";
      continue;
    }
    if (inTable) { html += "</table>"; inTable = false; }
    if (/^### /.test(ln)) html += `<h4>${inline(ln.slice(4))}</h4>`;
    else if (/^## /.test(ln)) html += `<h3>${inline(ln.slice(3))}</h3>`;
    else if (/^# /.test(ln)) html += `<h2>${inline(ln.slice(2))}</h2>`;
    else if (/^> /.test(ln)) html += `<p class="note">${inline(ln.slice(2))}</p>`;
    else if (/^- /.test(ln)) html += `<li>${inline(ln.slice(2))}</li>`;
    else if (/^---/.test(ln)) html += "<hr>";
    else if (ln.trim()) html += `<p>${inline(ln)}</p>`;
  }
  if (inTable) html += "</table>";
  return html;
  function inline(s) {
    return esc(s)
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/\*(.+?)\*/g, "<i>$1</i>")
      .replace(/`(.+?)`/g, "<code>$1</code>")
      .replace(/\[([^\]]+)\]\(([^)]+)\)/g,
               '<a href="$2" target="_blank">$1</a>');
  }
}

/* ── Review: inline scaffold + thought map ────────────────────────────── */
let reviewMapData = null;

$("#r-map-run").onclick = () => withProgress("Mapping your themes…",
  async () => {
    reviewMapData = await api("/api/review_map");
    $("#r-map-wrap").classList.remove("hidden");
    drawReviewMap();
  }).catch(e => alert("map failed: " + e.message));

function drawReviewMap() {
  const d = reviewMapData;
  const canvas = $("#r-map");
  canvas.width = canvas.parentElement.clientWidth - 20;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const ths = d.themes;
  if (!ths.length) return;
  // circular layout, biggest theme centred
  const sorted = [...ths].sort((a, b) => b.n - a.n);
  const maxN = sorted[0].n;
  const pos = {};
  sorted.forEach((t, i) => {
    if (i === 0) pos[t.id] = { x: W / 2, y: H / 2 };
    else {
      const ang = (i - 1) / (sorted.length - 1) * 2 * Math.PI - Math.PI / 2;
      pos[t.id] = { x: W / 2 + Math.cos(ang) * W * 0.33,
                    y: H / 2 + Math.sin(ang) * H * 0.36 };
    }
    t.r = 18 + 34 * Math.sqrt(t.n / maxN);
  });
  // links
  for (const l of d.links) {
    const a = pos[l.a], b = pos[l.b];
    if (!a || !b) continue;
    ctx.strokeStyle = `rgba(77,163,255,${0.12 + 0.5 * l.w})`;
    ctx.lineWidth = 1 + 5 * l.w;
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }
  // bubbles
  for (const t of ths) {
    const p = pos[t.id];
    ctx.fillStyle = "#1e2740";
    ctx.beginPath(); ctx.arc(p.x, p.y, t.r, 0, 7); ctx.fill();
    ctx.strokeStyle = t.gap ? "#ffb454" : "#2a3654";
    ctx.lineWidth = t.gap ? 3 : 1.5;
    ctx.beginPath(); ctx.arc(p.x, p.y, t.r, 0, 7); ctx.stroke();
    // green arc = screened share
    if (t.n) {
      ctx.strokeStyle = "#43d17c"; ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.arc(p.x, p.y, t.r + 4, -Math.PI / 2,
              -Math.PI / 2 + 2 * Math.PI * (t.rated / t.n));
      ctx.stroke();
    }
    ctx.fillStyle = "#dbe4f5"; ctx.font = "12px sans-serif";
    ctx.textAlign = "center";
    const words = t.label.split(" ");
    words.slice(0, 3).forEach((w, i) =>
      ctx.fillText(w.slice(0, 14), p.x, p.y - 6 + i * 13));
    ctx.fillStyle = "#8494b5";
    ctx.fillText(`${t.rated}/${t.n}`, p.x, p.y + t.r - 6);
  }
  canvas.onclick = (ev) => {
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const hit = ths.find(t => {
      const p = pos[t.id];
      return (p.x - x) ** 2 + (p.y - y) ** 2 < t.r ** 2;
    });
    const det = $("#r-map-detail");
    if (!hit) { det.classList.add("hidden"); return; }
    det.classList.remove("hidden");
    det.innerHTML = `<b>${esc(hit.label)}</b> — ${hit.n} papers,
      ${hit.rated} screened${hit.gap ?
        ' · <span class="warn">under-screened (gap)</span>' : ""}
      ${hit.years ? ` · ${hit.years[0]}–${hit.years[1]}` : ""}<br>` +
      hit.top.map(t2 => `<div class="res-meta">• ${esc(t2)}</div>`).join("");
  };
}

/* ── Evolution charts ─────────────────────────────────────────────────── */
function drawSeries(canvasSel, series, fields, colors, normalize) {
  const canvas = $(canvasSel);
  canvas.width = canvas.parentElement.clientWidth - 20;
  const W = canvas.width, H = canvas.height, pad = 34;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  if (series.length < 2) {
    ctx.fillStyle = "#8494b5"; ctx.font = "13px sans-serif";
    ctx.fillText("need at least 2 snapshots", 20, 30);
    return;
  }
  let max = 1;
  if (!normalize)
    for (const f of fields)
      for (const s of series) max = Math.max(max, +s[f] || 0);
  ctx.strokeStyle = "#2a3654";
  ctx.strokeRect(pad, 8, W - pad - 8, H - pad - 8);
  ctx.fillStyle = "#8494b5"; ctx.font = "11px sans-serif";
  ctx.fillText(normalize ? "1.0" : String(max), 4, 16);
  ctx.fillText("0", 4, H - pad);
  fields.forEach((f, fi) => {
    ctx.strokeStyle = colors[fi]; ctx.lineWidth = 2; ctx.beginPath();
    series.forEach((s, i) => {
      const v = +s[f] || 0;
      const x = pad + (W - pad - 8) * i / (series.length - 1);
      const y = 8 + (H - pad - 16) * (1 - v / (normalize ? 1 : max));
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
}
async function loadEvolution() {
  const d = await api("/api/evolution");
  $("#ev-empty").classList.toggle("hidden", d.series.length >= 2);
  drawSeries("#ev-growth", d.series,
    ["total_papers", "rated_papers", "niche_clusters"],
    ["#4da3ff", "#43d17c", "#c084fc"], false);
  drawSeries("#ev-edges", d.series,
    ["semantic_edges", "citation_edges"], ["#4da3ff", "#ffb454"], false);
  drawSeries("#ev-quality", d.series,
    ["modularity_combined", "clustering_combined", "largest_component_frac"],
    ["#4da3ff", "#43d17c", "#ffb454"], true);
}

/* ── Sentinel (surveillance) ──────────────────────────────────────────── */
async function loadSentinel() {
  try {
    const s = await api("/api/sentinel");
    $("#sn-enabled").checked = s.config.enabled;
    $("#sn-interval").value = s.config.interval_hours;
    $("#sn-thresh").value = s.config.min_score;
    $("#sn-dot").className = "dot " + (s.config.enabled ? "on" : "dim");
    const last = s.config.last_run
      ? new Date(s.config.last_run * 1000).toLocaleString() : "never";
    $("#sn-status").textContent =
      `last scan: ${last} · inbox: ${s.inbox_count}`;
    const inbox = await api("/api/sentinel/inbox");
    const el = $("#sn-inbox");
    el.innerHTML = inbox.length
      ? "<h4>Inbox — new papers awaiting your verdict</h4>" : "";
    inbox.forEach(e => {
      const d = document.createElement("div");
      d.className = "banner";
      const doi = e.doi
        ? ` · <a href="https://doi.org/${esc(e.doi)}" target="_blank">open</a>` : "";
      d.innerHTML = `<b>${esc(e.title)}</b>
        <span class="res-meta">(${e.year ?? "?"}) · score ${e.score} ·
        via "${esc(e.watch)}" · ${esc(e.found_at)}${doi}</span><br>
        <span class="res-abs">${esc(e.abstract || "")}</span>
        <div class="rate-row">
          <button class="sn-accept primary">Add to graph</button>
          <button class="sn-dismiss">Dismiss</button>
        </div>`;
      d.querySelector(".sn-accept").onclick = async () => {
        await api("/api/sentinel/inbox/accept", { token: e.token });
        loadSentinel(); loadStats();
      };
      d.querySelector(".sn-dismiss").onclick = async () => {
        await api("/api/sentinel/inbox/dismiss", { token: e.token });
        loadSentinel();
      };
      el.appendChild(d);
    });
  } catch (e) { $("#sn-status").textContent = "sentinel unavailable"; }
}
$("#sn-save").onclick = async () => {
  await api("/api/sentinel", {
    enabled: $("#sn-enabled").checked,
    interval_hours: +$("#sn-interval").value,
    min_score: +$("#sn-thresh").value });
  loadSentinel();
};
$("#sn-scan").onclick = async () => {
  $("#sn-scan").disabled = true;
  try {
    const r = await withProgress("Sentinel sweep — scanning watches…", () =>
      api("/api/sentinel/scan", {}));
    $("#sn-status").textContent =
      `${r.new} new paper(s) filed` +
      (r.digest ? ` · digest written` : "");
    loadSentinel();
  } catch (e) { $("#sn-status").textContent = "error: " + e.message; }
  finally { $("#sn-scan").disabled = false; }
};

/* ── Watches tab ──────────────────────────────────────────────────────── */
async function loadWatches() {
  const ws = await api("/api/watches");
  const el = $("#w-list");
  el.innerHTML = ws.length ? "" : '<p class="note">No watches yet.</p>';
  ws.forEach((w, i) => {
    const d = document.createElement("div");
    d.className = "banner";
    d.innerHTML = `<b>${esc(w.query)}</b> ` +
      (w.keywords?.length ? `+ [${esc(w.keywords.join(", "))}] ` : "") +
      `<span class="res-meta">last checked ${esc(w.last_checked)}</span> ` +
      `<button style="float:right">delete</button>`;
    d.querySelector("button").onclick = async () => {
      await api("/api/watches/delete", { index: i }); loadWatches(); };
    el.appendChild(d);
  });
}
$("#w-add").onclick = async () => {
  const q = $("#w-q").value.trim();
  if (!q) return;
  await api("/api/watches", { query: q, keywords: $("#w-kw").value });
  $("#w-q").value = ""; $("#w-kw").value = "";
  loadWatches();
};
$("#w-check").onclick = async () => {
  $("#w-status").textContent = "checking…";
  try {
    const reps = await api("/api/watches/check", {});
    const el = $("#w-results"); el.innerHTML = "";
    reps.forEach(rep => {
      const h = document.createElement("h3");
      h.textContent = `${rep.watch.query} — ${rep.results.length} new`;
      el.appendChild(h);
      rep.results.forEach(p => el.appendChild(resultCard(p)));
    });
    $("#w-status").textContent = "";
    loadWatches();
  } catch (e) { $("#w-status").textContent = "error: " + e.message; }
};

/* ── Collaborate tab (social-psyche) ──────────────────────────────────── */
async function loadCollab() {
  try {
    const id = await api("/api/sp/identity");
    $("#c-fp").textContent = id.fingerprint;
    $("#c-unavailable").classList.add("hidden");
    $("#c-body").classList.remove("hidden");
  } catch (e) {
    $("#c-unavailable").classList.remove("hidden");
    $("#c-body").classList.add("hidden");
    return;
  }
  try {
    const lg = await api("/api/sp/ledger");
    const b = lg.balance;
    $("#c-ledger").innerHTML =
      `<p>exchanges <b>${b.exchanges}</b> · contribution score ` +
      `<b class="ok">${b.contribution_score}</b> · received ` +
      `<b>${b.papers_received}</b> · peers <b>${b.distinct_peers}</b> · chain ` +
      (lg.verified ? '<b class="ok">verified</b>' : '<b class="err">BROKEN</b>') +
      "</p>" +
      (lg.entries.length ? "<table><tr><th>#</th><th>when</th><th>init gave</th>" +
        "<th>resp gave</th><th>shared</th></tr>" +
        lg.entries.map(e => `<tr><td>${e.height}</td><td>${esc(e.body.ts)}</td>` +
          `<td>${e.body.initiator_contributed}</td>` +
          `<td>${e.body.responder_contributed}</td>` +
          `<td>${e.body.shared_found}</td></tr>`).join("") + "</table>" : "");
  } catch (e) { $("#c-ledger").textContent = "ledger unavailable: " + e.message; }
  loadPeers();
}
async function loadPeers() {
  const ps = await api("/api/sp/peers");
  const el = $("#c-peers");
  const sel = $("#cm-peer");
  sel.innerHTML = '<option value="">— pinned peer —</option>';
  el.innerHTML = ps.length ? "" : '<p class="note">No pinned peers.</p>';
  ps.forEach(p => {
    const d = document.createElement("div");
    d.className = "banner";
    d.innerHTML = `<b>${esc(p.name)}</b> ${esc(p.host)}:${p.port}
      <code>${esc(p.fingerprint.slice(0, 24))}…</code>
      <button style="float:right">unpin</button>`;
    d.querySelector("button").onclick = async () => {
      await api("/api/sp/peers/delete", { name: p.name }); loadPeers(); };
    el.appendChild(d);
    const o = document.createElement("option");
    o.value = p.name; o.textContent = p.name;
    sel.appendChild(o);
  });
}
$("#cp-add").onclick = async () => {
  try {
    await api("/api/sp/peers", { name: $("#cp-name").value,
      host: $("#cp-host").value, port: +$("#cp-port").value,
      fingerprint: $("#cp-fp").value });
    loadPeers();
  } catch (e) { alert(e.message); }
};
function mergeReport(r) {
  return `peer <code>${esc(r.peer_fingerprint)}</code><br>
    shared (PSI) <b>${r.shared}</b> · imported <b class="ok">${r.imported}</b>
    · by-DOI ${r.shared_by_doi} · novel ${r.novel_regions}<br>
    jaccard ${fmt(r.jaccard_doi)} · spectral ${fmt(r.spectral_distance)} ·
    deltacon ${fmt(r.deltacon)} · GW ${fmt(r.gw_distortion)}` +
    (r.notes?.length ? `<br><span class="warn">${esc(r.notes.join(" · "))}</span>` : "");
}
const fmt = (x) => x == null ? "n/a" : (+x).toFixed(3);
$("#cm-run").onclick = async () => {
  const mode = $("#cm-mode").value;
  $("#cm-status").textContent = mode === "serve"
    ? "waiting for peer…" : "connecting…";
  $("#cm-run").disabled = true;
  const out = $("#cm-out");
  try {
    const r = await api("/api/sp/merge", { mode,
      peer: $("#cm-peer").value, host: $("#cm-host").value,
      port: +($("#cm-port").value || 9333),
      share_ids: $("#cm-ids").checked });
    if (r.status === "done") {
      out.classList.remove("hidden"); out.innerHTML = mergeReport(r.result);
      $("#cm-status").textContent = ""; loadCollab(); loadStats();
    } else {                              // serve mode: poll
      const poll = setInterval(async () => {
        const st = await api("/api/sp/merge/status");
        if (st.status === "done") {
          clearInterval(poll); out.classList.remove("hidden");
          out.innerHTML = mergeReport(st.result);
          $("#cm-status").textContent = ""; $("#cm-run").disabled = false;
          loadCollab(); loadStats();
        } else if (st.status === "error") {
          clearInterval(poll);
          $("#cm-status").textContent = "error: " + st.error;
          $("#cm-run").disabled = false;
        }
      }, 2000);
      return;                             // keep button disabled while waiting
    }
  } catch (e) { $("#cm-status").textContent = "error: " + e.message; }
  $("#cm-run").disabled = false;
};
$("#ce-run").onclick = async () => {
  try {
    const r = await api("/api/sp/export_capsule",
      { share_ids: $("#ce-ids").checked });
    const out = $("#ce-out");
    out.classList.remove("hidden");
    out.innerHTML = `capsule <code>${esc(r.capsule)}</code><br>
      signature <code>${esc(r.signature)}</code><br>
      ${r.stats.n_papers} papers — host both files anywhere; consumers verify
      offline with <code>social-psyche verify</code>.`;
  } catch (e) { alert(e.message); }
};

/* ── Services (Neo4j / GROBID / LLM) ──────────────────────────────────── */
let svcCache = null;

function svcChip(label, cls) {
  return `<span class="svc-chip"><i class="dot ${cls}"></i>${label}</span>`;
}
async function pollServices() {
  try {
    svcCache = await api("/api/services");
    const s = svcCache;
    $("#svcbar").innerHTML =
      svcChip("Neo4j", s.neo4j.bolt ? "on" : (s.neo4j.http ? "dim" : "off")) +
      svcChip("GROBID", s.grobid.alive ? "on" : "off") +
      svcChip("LLM", !s.llm.enabled ? "dim" : (s.llm.available ? "on" : "off")) +
      svcChip(s.backend, "dim");
    if (document.querySelector("#tab-services.active")) renderServices();
  } catch (e) { /* server briefly busy — keep last chips */ }
}
$("#svcbar").onclick = () => showTab("services");

function renderServices() {
  const s = svcCache;
  if (!s) return;
  $("#sv-docker").classList.toggle("hidden", s.docker);
  const nd = $("#sv-neo4j-dot");
  nd.className = "dot " + (s.neo4j.bolt ? "on" : (s.neo4j.http ? "dim" : "off"));
  $("#sv-neo4j-info").textContent = s.neo4j.bolt
    ? "running — bolt authenticated, usable as backend"
    : s.neo4j.http
      ? "HTTP up but bolt failed: " + (s.neo4j.reason || "auth?")
      : "not running";
  $("#sv-grobid-dot").className = "dot " + (s.grobid.alive ? "on" : "off");
  $("#sv-grobid-status").textContent = s.grobid.alive
    ? "running — PDF imports get full extraction" : "";
  $("#sv-llm-dot").className =
    "dot " + (!s.llm.enabled ? "dim" : (s.llm.available ? "on" : "off"));
  $("#sv-llm-info").textContent = s.llm.available
    ? `connected — model ${s.llm.model}`
    : (s.llm.enabled ? (s.llm.error || "unreachable") : "disabled");
  $("#sv-backend").textContent = s.backend;
  $("#sv-neo4j-use").disabled = s.backend === "Neo4j";
  $("#sv-neo4j-use").textContent =
    s.backend === "Neo4j" ? "Active backend ✓" : "Use as backend now";
}
async function loadServices() {
  await pollServices(); renderServices();
  try {
    const k = await api("/api/core_key");
    $("#sv-core-dot").className = "dot " + (k.set ? "on" : "dim");
    $("#sv-core-status").textContent = k.set
      ? "key set — fast lane active" : "no key — polite (slow) rate";
  } catch (e) {}
}
$("#sv-core-save").onclick = async () => {
  try {
    const r = await api("/api/core_key", { key: $("#sv-core-key").value });
    $("#sv-core-key").value = "";
    $("#sv-core-dot").className = "dot " + (r.set ? "on" : "dim");
    $("#sv-core-status").textContent = r.set
      ? "key saved — fast lane active" : "key cleared";
  } catch (e) { $("#sv-core-status").textContent = "error: " + e.message; }
};
$("#sv-core-test").onclick = async () => {
  $("#sv-core-status").textContent = "testing…";
  try {
    const r = await api("/api/core_test", {});
    $("#sv-core-status").innerHTML =
      `<span class="${r.ok ? "ok" : "err"}">${esc(r.detail)}</span>`;
    $("#sv-core-dot").className =
      "dot " + (r.ok && r.has_key ? "on" : (r.ok ? "dim" : "off"));
  } catch (e) { $("#sv-core-status").textContent = "error: " + e.message; }
};
$("#sv-core-enrich").onclick = async () => {
  $("#sv-core-enrich").disabled = true;
  $("#sv-core-enrich-status").textContent = "";
  try {
    const r = await withProgress("Asking CORE for full texts…", () =>
      api("/api/core_enrich",
          { retry_failed: $("#sv-core-retry").checked }));
    $("#sv-core-enrich-status").innerHTML =
      `<span class="ok">${r.enriched} enriched</span> of ${r.checked} tried`;
    loadStats();
  } catch (e) {
    $("#sv-core-enrich-status").textContent = "error: " + e.message;
  } finally { $("#sv-core-enrich").disabled = false; }
};

async function svcAction(name, action, statusEl) {
  $(statusEl).textContent = action + "ing… (docker can take ~30 s)";
  try {
    const r = await api(`/api/services/${action}`, { name });
    $(statusEl).textContent = r.error ? "error: " + r.error :
      (r.already_running ? "already running" : "done");
  } catch (e) { $(statusEl).textContent = "error: " + e.message; }
  await pollServices(); renderServices();
}
$("#sv-neo4j-start").onclick = () => svcAction("neo4j", "start", "#sv-neo4j-status");
$("#sv-neo4j-stop").onclick = () => svcAction("neo4j", "stop", "#sv-neo4j-status");
$("#sv-grobid-start").onclick = () => svcAction("grobid", "start", "#sv-grobid-status");
$("#sv-grobid-stop").onclick = () => svcAction("grobid", "stop", "#sv-grobid-status");

$("#sv-neo4j-use").onclick = async () => {
  const st = $("#sv-neo4j-status");
  st.textContent = "starting Neo4j + switching backend…";
  $("#sv-neo4j-use").disabled = true;
  try {
    const r = await api("/api/services/start", { name: "neo4j" });
    if (r.error) throw new Error(r.error);
    const sw = await api("/api/switch_backend", {});
    st.textContent = "backend: " + sw.backend;
    loadStats();
  } catch (e) { st.textContent = "error: " + e.message; }
  await pollServices(); renderServices();
};

/* ── boot ─────────────────────────────────────────────────────────────── */
loadStats();
loadGraph();
pollServices();
setInterval(pollServices, 20000);
