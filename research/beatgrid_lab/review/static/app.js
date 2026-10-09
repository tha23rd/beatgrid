// beatgrid listen: play a track with clicks on its grids and see them over the waveform.
"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = { reference: "#22c55e", beatgrid: "#f59e0b", engine: "#94a3b8", "beatthis-raw": "#c084fc" };
const LOOKAHEAD = 0.2; // seconds of clicks scheduled ahead of the audio clock

const state = {
  set: null, rows: [], track: null, buffer: null, mono: null, rate: 44100,
  bands: null, bandsDt: 0.005, flux: null, fluxDt: 0, fluxT0: 0,
  view: { start: 0, span: 8 }, playing: false, ctx: null, source: null,
  t0: 0, offset: 0, scheduled: {}, timer: null, musicGain: null, clickGain: null, clicks: null,
  disagreements: [],
};

// ------------------------------------------------------------------ data

async function json(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function loadSets() {
  const sets = await json("/api/sets");
  $("set").innerHTML = sets.map((s) => `<option value="${s.name}">${s.name} (${s.tracks})</option>`).join("");
  const params = new URLSearchParams(location.hash.slice(1));
  if (params.get("set")) $("set").value = params.get("set");
  await loadTracks(params.get("id"));
}

async function loadTracks(selectId) {
  state.set = $("set").value;
  $("list").innerHTML = `<li class="muted">scoring every track… (first load of a set takes a minute)</li>`;
  const data = await json(`/api/tracks?set=${encodeURIComponent(state.set)}`);
  state.rows = data.rows;
  renderList();
  if (selectId) selectTrack(selectId);
}

function rowMatches(r) {
  const f = $("filter").value, q = $("search").value.toLowerCase(), split = $("split").value;
  if (split && r.split !== split) return false;
  if (q && !`${r.title} ${r.artist || ""}`.toLowerCase().includes(q)) return false;
  const s = r.scores.beatgrid;
  if (f === "fail") return !s.usable_10;
  if (f === "disagree") return !(s.within_10 >= 0.9 && s.bar_ok && s.octave === "1x");
  if (f === "review") return r.review.length > 0;
  if (f === "drifts") return r.audit === "drifts";
  if (f === "unjudged") return !r.verdict;
  return true;
}

function badge(text, cls) { return `<span class="badge ${cls}">${text}</span>`; }

function renderList() {
  const rows = state.rows.filter(rowMatches);
  $("count").textContent = `${rows.length} of ${state.rows.length}`;
  $("list").innerHTML = rows.map((r) => {
    const s = r.scores.beatgrid;
    const b = [];
    b.push(s.usable_10 ? badge("usable", "good") : badge(s.octave !== "1x" ? `octave ${s.octave}` : !s.bar_ok ? "bar" : `${Math.round(s.within_10 * 100)}% ≤10ms`, "bad"));
    if (r.audit === "drifts") b.push(badge("ref drifts", "warn"));
    if (r.review.length) b.push(badge("review", "warn"));
    if (r.verdict) b.push(badge(`✓ ${r.verdict}`, "info"));
    const who = r.artist ? `${r.artist} – ` : "";
    return `<li data-id="${r.id}" class="${state.track && state.track.id === r.id ? "active" : ""}">
      <div class="t">${who}${r.title}</div>
      <div class="m"><span>${r.ref_bpm.toFixed(2)} BPM</span><span>${r.kind}/${r.split}</span>${b.join("")}</div></li>`;
  }).join("");
}

async function selectTrack(id) {
  stop();
  location.hash = `set=${encodeURIComponent(state.set)}&id=${encodeURIComponent(id)}`;
  $("placeholder").textContent = "loading…";
  $("content").hidden = true;
  $("track").classList.remove("empty");
  const q = `set=${encodeURIComponent(state.set)}&id=${encodeURIComponent(id)}`;
  const [track, audio, bands, flux] = await Promise.all([
    json(`/api/track?${q}`),
    fetch(`/api/audio?${q}`).then((r) => r.arrayBuffer()),
    fetch(`/api/bands?${q}`).then(async (r) => ({ dt: +r.headers.get("X-Dt"), data: new Float32Array(await r.arrayBuffer()) })),
    fetch(`/api/flux?${q}`).then(async (r) => ({ dt: +r.headers.get("X-Dt"), t0: +r.headers.get("X-T0"), data: new Float32Array(await r.arrayBuffer()) })),
  ]);
  ensureAudio();
  state.track = track;
  state.buffer = await state.ctx.decodeAudioData(audio);
  state.rate = state.buffer.sampleRate;
  state.mono = mixdown(state.buffer);
  const n = bands.data.length / 3;
  state.bands = [bands.data.subarray(0, n), bands.data.subarray(n, 2 * n), bands.data.subarray(2 * n)];
  state.bandsDt = bands.dt;
  state.flux = flux.data; state.fluxDt = flux.dt; state.fluxT0 = flux.t0;
  state.bandMax = [0, 1, 2].map((i) => percentile(state.bands[i], 0.99) || 1);
  state.fluxMax = percentile(state.flux, 0.999) || 1;
  for (const grid of Object.values(track.grids)) {
    if (!grid) continue;
    // Bar number of each beat, counted from the first bar line in the track.
    let bar = 0;
    grid.bars = grid.downbeat.map((d) => (d ? ++bar : bar));
  }
  state.disagreements = findDisagreements();
  state.offset = Math.max(0, track.music[0]);
  centreOn(state.offset);
  renderTrack();
  renderList();
  $("placeholder").textContent = "";
  $("content").hidden = false;
  resizeCanvases();
  draw();
}

function mixdown(buffer) {
  const out = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const d = buffer.getChannelData(c);
    for (let i = 0; i < d.length; i++) out[i] += d[i] / buffer.numberOfChannels;
  }
  return out;
}

function percentile(arr, p) {
  const step = Math.max(1, Math.floor(arr.length / 20000));
  const sample = [];
  for (let i = 0; i < arr.length; i += step) sample.push(arr[i]);
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.floor(p * sample.length))];
}

// Where beatgrid and the reference disagree: beats more than 10 ms apart, or bar lines
// on different beats.
function findDisagreements() {
  const ref = state.track.grids.reference, ours = state.track.grids.beatgrid;
  if (!ref || !ours) return [];
  const out = [];
  let j = 0;
  for (let i = 0; i < ref.times.length; i++) {
    const t = ref.times[i];
    if (t < state.track.music[0] || t > state.track.music[1]) continue;
    while (j + 1 < ours.times.length && Math.abs(ours.times[j + 1] - t) <= Math.abs(ours.times[j] - t)) j++;
    const off = ours.times[j] - t;
    if (Math.abs(off) > 0.010 || ref.downbeat[i] !== ours.downbeat[j]) out.push(t);
  }
  return out;
}

// ------------------------------------------------------------------ rendering text

function renderTrack() {
  const t = state.track, m = t.meta;
  $("title").textContent = `${m.artist ? m.artist + " – " : ""}${m.title || t.id}`;
  $("subtitle").textContent = `${t.id} · ${t.kind} · ${t.split}${m.genre ? " · " + m.genre : ""} · ${fmt(t.duration)}`;
  const s = t.scores.beatgrid, a = t.audit;
  const b = [s.usable_10 ? badge("beatgrid usable @10 ms", "good") : badge("beatgrid not usable @10 ms", "bad")];
  if (a) b.push(badge(`reference ${a.verdict}${a.drift_ms != null ? ` (${a.drift_ms.toFixed(0)} ms)` : ""}`, a.verdict === "steady" ? "good" : "warn"));
  for (const reason of t.diagnostics.beatgrid.review || []) b.push(badge(reason, "warn"));
  $("badges").innerHTML = b.join("");
  const rows = Object.entries(t.scores).map(([k, s]) => `<tr><td style="color:${COLORS[k] || "inherit"}">${k}</td>
    <td>${s.bpm.toFixed(3)}</td><td>${s.octave}</td><td>${s.bar_ok ? "yes" : "no"}</td>
    <td>${(s.within_10 * 100).toFixed(0)}%</td><td>${(s.within_20 * 100).toFixed(0)}%</td>
    <td>${s.bias_ms != null ? s.bias_ms.toFixed(1) : "–"}</td><td>${s.usable_10 ? "✓" : "✗"}</td></tr>`);
  $("scores").innerHTML = `<tr><th>vs reference (${t.grids.reference.bpm.toFixed(3)} BPM)</th><th>BPM</th><th>octave</th><th>bar 1</th><th>≤10 ms</th><th>≤20 ms</th><th>bias ms</th><th>usable</th></tr>${rows.join("")}`;
  const d = t.diagnostics.beatgrid;
  $("diag").textContent = Object.entries(d).filter(([k]) => k !== "review")
    .map(([k, v]) => `${k}=${typeof v === "number" ? +v.toFixed(4) : v}`).join("  ");
  for (const btn of document.querySelectorAll(".verdict button")) btn.classList.toggle("chosen", t.verdict && t.verdict.verdict === btn.dataset.v);
  $("note").value = (t.verdict && t.verdict.note) || "";
  $("saved").textContent = t.verdict ? `saved ${t.verdict.at}` : "";
}

function fmt(s) {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(3).padStart(6, "0")}`;
}

// ------------------------------------------------------------------ audio

function ensureAudio() {
  if (state.ctx) return;
  state.ctx = new AudioContext({ latencyHint: "interactive" });
  state.musicGain = state.ctx.createGain();
  state.clickGain = state.ctx.createGain();
  state.musicGain.connect(state.ctx.destination);
  state.clickGain.connect(state.ctx.destination);
  state.musicGain.gain.value = +$("musicvol").value;
  state.clickGain.gain.value = +$("clickvol").value;
  state.clicks = { beat: clickBuffer(1800, 0.5), down: clickBuffer(2800, 0.9) };
}

function clickBuffer(freq, level) {
  const sr = state.ctx.sampleRate, n = Math.round(sr * 0.012);
  const b = state.ctx.createBuffer(1, n, sr), d = b.getChannelData(0);
  for (let i = 0; i < n; i++) d[i] = level * Math.sin((2 * Math.PI * freq * i) / sr) * Math.exp(-i / (sr * 0.0025));
  return b;
}

function now() {
  // Track time at the audio clock; for drawing, subtract the output latency so the
  // playhead matches what is heard.
  if (!state.playing) return state.offset;
  return state.offset + (state.ctx.currentTime - state.t0);
}

function heardNow() {
  if (!state.playing) return state.offset;
  return now() - (state.ctx.outputLatency || state.ctx.baseLatency || 0);
}

function play() {
  if (!state.buffer) return;
  ensureAudio();
  state.ctx.resume();
  stopSource();
  const src = state.ctx.createBufferSource();
  src.buffer = state.buffer;
  src.connect(state.musicGain);
  state.t0 = state.ctx.currentTime + 0.05;
  src.start(state.t0, state.offset);
  src.onended = () => { if (state.source === src) { state.playing = false; $("play").textContent = "▶ play"; } };
  state.source = src;
  state.playing = true;
  state.scheduled = {};
  $("play").textContent = "❚❚ pause";
  state.timer = setInterval(scheduleClicks, 25);
  scheduleClicks();
  requestAnimationFrame(loop);
}

function stopSource() {
  if (state.source) { state.source.onended = null; try { state.source.stop(); } catch (e) {} state.source = null; }
  clearInterval(state.timer);
}

function stop() {
  if (state.playing) state.offset = now();
  state.playing = false;
  stopSource();
  $("play").textContent = "▶ play";
}

function toggle() { state.playing ? stop() : play(); }

function seek(t) {
  const playing = state.playing;
  stop();
  state.offset = Math.max(0, Math.min(t, state.buffer ? state.buffer.duration : t));
  if (playing) play(); else draw();
}

function clickGrids() {
  const mode = $("clickmode").value;
  if (mode === "off") return [];
  if (mode === "split") return [["reference", -0.8], ["beatgrid", 0.8]];
  return [[mode, 0]];
}

function scheduleClicks() {
  if (!state.playing) return;
  const until = now() + LOOKAHEAD;
  for (const [name, pan] of clickGrids()) {
    const g = state.track.grids[name];
    if (!g) continue;
    let from = state.scheduled[name] ?? state.offset - 1e-6;
    let i = lowerBound(g.times, from + 1e-6);
    for (; i < g.times.length && g.times[i] <= until; i++) {
      const at = state.t0 + (g.times[i] - state.offset);
      if (at < state.ctx.currentTime) continue;
      const s = state.ctx.createBufferSource();
      s.buffer = g.downbeat[i] ? state.clicks.down : state.clicks.beat;
      const p = state.ctx.createStereoPanner();
      p.pan.value = pan;
      s.connect(p).connect(state.clickGain);
      s.start(at);
      from = g.times[i];
    }
    state.scheduled[name] = from;
  }
}

function lowerBound(arr, x) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid] < x) lo = mid + 1; else hi = mid; }
  return lo;
}

// ------------------------------------------------------------------ drawing

function resizeCanvases() {
  for (const c of [$("overview"), $("detail")]) {
    const w = c.clientWidth * devicePixelRatio;
    c.width = w;
    c.height = (c.id === "overview" ? 70 : 300) * devicePixelRatio;
  }
}

function centreOn(t) {
  state.view.start = Math.max(0, t - state.view.span / 3);
}

function loop() {
  if (!state.playing) return draw();
  const t = heardNow();
  if (t > state.view.start + state.view.span * 0.75 || t < state.view.start) centreOn(t);
  draw();
  requestAnimationFrame(loop);
}

function bandColour(t, alpha = 1) {
  const i = Math.min(state.bands[0].length - 1, Math.max(0, Math.floor(t / state.bandsDt)));
  const l = state.bands[0][i] / state.bandMax[0], m = state.bands[1][i] / state.bandMax[1], h = state.bands[2][i] / state.bandMax[2];
  const k = 255 / Math.max(l, m, h, 1e-6);
  // rekordbox-style: red lows, green mids, blue highs
  return `rgba(${Math.round(60 + l * k * 0.75)},${Math.round(60 + m * k * 0.7)},${Math.round(70 + h * k * 0.75)},${alpha})`;
}

function draw() {
  if (!state.buffer) return;
  drawOverview();
  drawDetail();
  drawReadout();
}

function drawOverview() {
  const c = $("overview"), g = c.getContext("2d"), W = c.width, H = c.height;
  const dur = state.buffer.duration;
  g.clearRect(0, 0, W, H);
  const low = state.bands[0], dt = state.bandsDt;
  for (let x = 0; x < W; x++) {
    const a = Math.floor((x / W) * dur / dt), b = Math.max(a + 1, Math.floor(((x + 1) / W) * dur / dt));
    let e = 0;
    for (let i = a; i < b && i < low.length; i++) e = Math.max(e, state.bands[0][i] / state.bandMax[0], state.bands[1][i] / state.bandMax[1], state.bands[2][i] / state.bandMax[2]);
    const h = Math.min(1, e) * (H - 14);
    g.fillStyle = bandColour((x / W) * dur);
    g.fillRect(x, (H - 14 - h) / 2 + 2, 1, h);
  }
  // disagreement ticks along the bottom
  g.fillStyle = "#ef4444";
  for (const t of state.disagreements) g.fillRect((t / dur) * W, H - 8, Math.max(1, devicePixelRatio), 6);
  // music span
  g.fillStyle = "rgba(255,255,255,0.06)";
  g.fillRect(0, 0, (state.track.music[0] / dur) * W, H);
  g.fillRect((state.track.music[1] / dur) * W, 0, W, H);
  // viewport + playhead
  g.strokeStyle = "rgba(255,255,255,0.7)";
  g.lineWidth = devicePixelRatio;
  g.strokeRect((state.view.start / dur) * W, 1, (state.view.span / dur) * W, H - 2);
  g.fillStyle = "#fff";
  g.fillRect((heardNow() / dur) * W, 0, devicePixelRatio, H);
}

function drawDetail() {
  const c = $("detail"), g = c.getContext("2d"), W = c.width, H = c.height;
  const { start, span } = state.view, rate = state.rate, mono = state.mono;
  const px = (t) => ((t - start) / span) * W;
  g.clearRect(0, 0, W, H);
  // waveform: sample min/max per column, coloured by band balance
  const mid = H / 2, amp = H * 0.46;
  for (let x = 0; x < W; x++) {
    const t = start + (x / W) * span;
    const a = Math.max(0, Math.floor(t * rate)), b = Math.min(mono.length, Math.max(a + 1, Math.floor((start + ((x + 1) / W) * span) * rate)));
    let lo = 0, hi = 0;
    for (let i = a; i < b; i++) { const v = mono[i]; if (v < lo) lo = v; if (v > hi) hi = v; }
    g.fillStyle = bandColour(t, 0.9);
    g.fillRect(x, mid - hi * amp, 1, Math.max(1, (hi - lo) * amp));
  }
  // attack curve (the flux beatgrid places grid lines with)
  if ($("showflux").checked && state.flux) {
    g.strokeStyle = "rgba(255,255,255,0.55)";
    g.lineWidth = devicePixelRatio;
    g.beginPath();
    const base = H - 4, scale = H * 0.22 / state.fluxMax;
    for (let x = 0; x < W; x++) {
      const t = start + (x / W) * span;
      const a = Math.floor((t - state.fluxT0) / state.fluxDt), b = Math.floor((start + ((x + 1) / W) * span - state.fluxT0) / state.fluxDt);
      let v = 0;
      for (let i = Math.max(0, a); i <= Math.max(a, b) && i < state.flux.length; i++) v = Math.max(v, state.flux[i]);
      const y = base - Math.min(H * 0.24, v * scale);
      x === 0 ? g.moveTo(x, y) : g.lineTo(x, y);
    }
    g.stroke();
  }
  // grids: reference in the top half, beatgrid in the bottom half, the others as strips
  const lanes = { reference: [0, mid], beatgrid: [mid, H], engine: [0, 14 * devicePixelRatio], "beatthis-raw": [H - 14 * devicePixelRatio, H] };
  for (const [name, [y0, y1]] of Object.entries(lanes)) {
    const grid = state.track.grids[name];
    if (!grid) continue;
    const i0 = lowerBound(grid.times, start), colour = COLORS[name];
    for (let i = i0; i < grid.times.length && grid.times[i] <= start + span; i++) {
      const x = px(grid.times[i]);
      const down = grid.downbeat[i];
      g.fillStyle = colour;
      g.globalAlpha = name === "engine" || name === "beatthis-raw" ? 0.8 : down ? 1 : 0.75;
      const w = (down ? 3 : 1.5) * devicePixelRatio;
      g.fillRect(x - w / 2, y0, w, y1 - y0);
      if (down && (name === "reference" || name === "beatgrid") && span < 30) {
        g.globalAlpha = 1;
        g.font = `${11 * devicePixelRatio}px ui-monospace, monospace`;
        g.fillText(String(grid.bars[i]), x + 4 * devicePixelRatio, name === "reference" ? y0 + 26 * devicePixelRatio : y1 - 6 * devicePixelRatio);
      }
    }
    g.globalAlpha = 1;
  }
  g.fillStyle = "rgba(255,255,255,0.15)";
  g.fillRect(0, mid, W, devicePixelRatio);
  // playhead
  const t = heardNow();
  g.fillStyle = "#fff";
  g.fillRect(px(t) - devicePixelRatio / 2, 0, devicePixelRatio, H);
}

function nearestIndex(times, t) {
  const i = lowerBound(times, t);
  if (i === 0) return 0;
  if (i >= times.length) return times.length - 1;
  return t - times[i - 1] <= times[i] - t ? i - 1 : i;
}

function drawReadout() {
  const t = heardNow(), ref = state.track.grids.reference, ours = state.track.grids.beatgrid;
  let text = `${fmt(t)}`;
  if (ref && ours) {
    const i = nearestIndex(ref.times, t), j = nearestIndex(ours.times, ref.times[i]);
    const off = (ours.times[j] - ref.times[i]) * 1000;
    const bar = ref.downbeat[i] === ours.downbeat[j] ? "same beat of the bar" : "different beat of the bar";
    text += `   beatgrid − reference at this beat: ${off >= 0 ? "+" : ""}${off.toFixed(1)} ms, ${bar}`;
    text += `   ·   ${state.disagreements.length} disagreeing beats in the music`;
  }
  $("readout").textContent = text;
}

// ------------------------------------------------------------------ interaction

function stepBeats(n) {
  const ref = state.track.grids.reference || state.track.grids.beatgrid;
  const i = nearestIndex(ref.times, now());
  const j = Math.max(0, Math.min(ref.times.length - 1, i + n));
  seek(ref.times[j] - 0.15);
  if (!state.playing) { centreOn(state.offset); draw(); }
}

function jumpDisagreement(dir) {
  const t = now() + 0.2 * dir;
  const list = state.disagreements;
  const target = dir > 0 ? list.find((x) => x > t + 0.05) : [...list].reverse().find((x) => x < t - 0.25);
  if (target != null) { seek(target - 1.0); centreOn(target); draw(); }
}

function setZoomFromSlider() {
  const v = +$("zoom").value; // 0..1 -> 60 s .. 0.08 s
  const span = 60 * Math.pow(0.08 / 60, v);
  const t = heardNow();
  const inView = t >= state.view.start && t <= state.view.start + state.view.span;
  const centre = state.playing || inView ? t : state.view.start + state.view.span / 2;
  state.view.span = span;
  state.view.start = Math.max(0, centre - span / 2);
  draw();
}

function wire() {
  $("set").onchange = () => loadTracks();
  for (const id of ["filter", "split"]) $(id).onchange = renderList;
  $("search").oninput = renderList;
  $("list").onclick = (e) => { const li = e.target.closest("li[data-id]"); if (li) selectTrack(li.dataset.id); };
  $("play").onclick = toggle;
  $("prevbar").onclick = () => stepBeats(-4);
  $("nextbar").onclick = () => stepBeats(4);
  $("prevdis").onclick = () => jumpDisagreement(-1);
  $("nextdis").onclick = () => jumpDisagreement(1);
  $("clickvol").oninput = (e) => state.clickGain && (state.clickGain.gain.value = +e.target.value);
  $("musicvol").oninput = (e) => state.musicGain && (state.musicGain.gain.value = +e.target.value);
  $("clickmode").onchange = () => { state.scheduled = {}; };
  $("zoom").oninput = setZoomFromSlider;
  $("showflux").onchange = draw;
  $("overview").onclick = (e) => {
    const r = e.target.getBoundingClientRect();
    const t = ((e.clientX - r.left) / r.width) * state.buffer.duration;
    seek(t); centreOn(t); draw();
  };
  $("detail").onclick = (e) => {
    const r = e.target.getBoundingClientRect();
    seek(state.view.start + ((e.clientX - r.left) / r.width) * state.view.span);
  };
  $("detail").onwheel = (e) => {
    e.preventDefault();
    const r = e.target.getBoundingClientRect();
    const at = state.view.start + ((e.clientX - r.left) / r.width) * state.view.span;
    const factor = Math.exp(e.deltaY * 0.0015);
    const span = Math.min(120, Math.max(0.05, state.view.span * factor));
    state.view.start = Math.max(0, at - ((at - state.view.start) / state.view.span) * span);
    state.view.span = span;
    $("zoom").value = Math.log(span / 60) / Math.log(0.08 / 60);
    draw();
  };
  for (const btn of document.querySelectorAll(".verdict button")) {
    btn.onclick = async () => {
      const t = state.track;
      const v = await json("/api/verdict", { method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ set: state.set, id: t.id, verdict: btn.dataset.v, note: $("note").value, time: now() }) });
      t.verdict = v;
      const row = state.rows.find((r) => r.id === t.id);
      if (row) row.verdict = v.verdict;
      renderTrack();
      renderList();
    };
  }
  window.onresize = () => { resizeCanvases(); draw(); };
  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" && e.target.type !== "range") return;
    if (!state.track) return;
    if (e.code === "Space") { e.preventDefault(); toggle(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); stepBeats(e.shiftKey ? 4 : 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); stepBeats(e.shiftKey ? -4 : -1); }
    else if (e.key === "n") jumpDisagreement(1);
    else if (e.key === "p") jumpDisagreement(-1);
    else if (e.key === "c") { const s = $("clickmode"); s.selectedIndex = (s.selectedIndex + 1) % s.options.length; state.scheduled = {}; }
    else if (e.key === "+" || e.key === "=") { $("zoom").value = Math.min(1, +$("zoom").value + 0.05); setZoomFromSlider(); }
    else if (e.key === "-") { $("zoom").value = Math.max(0, +$("zoom").value - 0.05); setZoomFromSlider(); }
  });
  setZoomFromSlider();
}

wire();
loadSets().catch((e) => { $("list").innerHTML = `<li class="muted">${e}</li>`; });
