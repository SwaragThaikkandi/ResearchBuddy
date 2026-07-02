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

/* ── Tabs ─────────────────────────────────────────────────────────────── */
document.querySelectorAll("#tabs button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("#tabs button").forEach(b =>
      b.classList.toggle("active", b === btn));
    document.querySelectorAll(".tab").forEach(t =>
      t.classList.toggle("active", t.id === "tab-" + btn.dataset.tab));
    if (btn.dataset.tab === "graph") loadGraph();
    if (btn.dataset.tab === "watches") loadWatches();
    if (btn.dataset.tab === "collab") loadCollab();
  };
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
  for (let i = 1; i <= 10; i++) {
    const b = document.createElement("button");
    b.textContent = i;
    b.onclick = async () => {
      try {
        await api("/api/rate", { token: p.token, rating: i });
        row.querySelectorAll("button").forEach(x => x.classList.remove("rated"));
        b.classList.add("rated");
        loadStats();
      } catch (e) { alert("rate failed: " + e.message); }
    };
    row.appendChild(b);
  }
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
  $("#d-status").textContent = "searching… (20–40 s)";
  $("#d-run").disabled = true;
  try {
    const r = await api("/api/search", {
      intent: $("#d-intent").value, keywords: $("#d-keywords").value,
      focus_ids: [...focusIds.keys()], n: 10 });
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
  $("#s-status").textContent = "snowballing…";
  $("#s-run").disabled = true;
  try {
    const r = await api("/api/snowball", { directions: dirs, n: 10,
      reset_frontier: $("#s-reset").checked });
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
  $("#h-status").textContent = "harvesting… (network)";
  $("#h-run").disabled = true;
  try {
    const r = await api("/api/harvest", { n: +$("#h-n").value });
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
  $("#r-status").textContent = "building pack…";
  $("#r-run").disabled = true;
  try {
    const r = await api("/api/review_pack", { use_llm: $("#r-llm").checked });
    const out = $("#r-out");
    out.classList.remove("hidden");
    out.innerHTML = `Pack written to <code>${esc(r.path)}</code><br>` +
      r.files.map(f => `<span class="res-tag">${esc(f)}</span>`).join(" ");
    $("#r-status").textContent = "";
  } catch (e) { $("#r-status").textContent = "error: " + e.message; }
  finally { $("#r-run").disabled = false; }
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

/* ── boot ─────────────────────────────────────────────────────────────── */
loadStats();
loadGraph();
