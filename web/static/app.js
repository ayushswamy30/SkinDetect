"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (p, digits = 0) => `${(p * 100).toFixed(digits)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// Counts are from the HAM10000 metadata (10,015 images). Order = model class order.
const CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
const LESIONS = {
  mel: { name: "Melanoma", n: 1113, malignant: true,
    text: "Cancer of the pigment cells. Often asymmetric, with several colours and an irregular network." },
  nv: { name: "Melanocytic nevus", n: 6705, malignant: false,
    text: "The common mole. Usually symmetric, with an even pigment network." },
  bcc: { name: "Basal cell carcinoma", n: 514, malignant: true,
    text: "The most common skin cancer. Grows slowly and rarely spreads; branching vessels are typical." },
  akiec: { name: "Actinic keratosis", n: 327, malignant: true,
    text: "Sun-damage lesions and intraepithelial carcinoma, which can progress to squamous cell carcinoma." },
  bkl: { name: "Benign keratosis", n: 1099, malignant: false,
    text: "Seborrhoeic keratoses and solar lentigines: harmless, but they can imitate melanoma." },
  df: { name: "Dermatofibroma", n: 115, malignant: false,
    text: "A firm, benign nodule, often a white centre ringed by a fine pigment network." },
  vasc: { name: "Vascular lesion", n: 142, malignant: false,
    text: "Cherry angiomas, angiokeratomas and pyogenic granulomas with red-purple lacunae." },
};
const CARD_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"];
const TOTAL = Object.values(LESIONS).reduce((s, l) => s + l.n, 0);

/* ---------- Nav, menu, scroll spy ---------- */
const nav = $("#nav");
const menu = $("#menu");
const toggle = $("#menuToggle");
const onScroll = () => nav.classList.toggle("is-solid", window.scrollY > 40);
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

function setMenu(open) {
  menu.hidden = !open;
  nav.classList.toggle("menu-open", open);
  toggle.setAttribute("aria-expanded", String(open));
  toggle.textContent = open ? "Close" : "Menu";
  document.body.style.overflow = open ? "hidden" : "";
}
toggle.addEventListener("click", () => setMenu(menu.hidden));
$$("#menu a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) setMenu(false); });

const navLinks = $$(".nav__links a");
const spy = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (!e.isIntersecting) continue;
    navLinks.forEach((a) => a.classList.toggle("is-current", a.getAttribute("href") === `#${e.target.id}`));
  }
}, { rootMargin: "-45% 0px -50% 0px" });
navLinks.forEach((a) => { const s = $(a.getAttribute("href")); if (s) spy.observe(s); });

/* ---------- Reveal on scroll & counters ---------- */
function animateCount(el) {
  const to = Number(el.dataset.to);
  if (reduceMotion) { el.textContent = to.toLocaleString(); return; }
  const start = performance.now();
  const tick = (now) => {
    const t = Math.min(1, (now - start) / 1600);
    el.textContent = Math.round(to * (1 - Math.pow(1 - t, 3))).toLocaleString();
    if (t < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
const io = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    entry.target.classList.add("is-in");
    $$(".count", entry.target).forEach(animateCount);
    io.unobserve(entry.target);
  }
}, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
$$(".reveal").forEach((el) => io.observe(el));

/* ---------- Lesion cards ---------- */
const carousel = $("#lesionCarousel");
carousel.innerHTML = CARD_ORDER.map((code) => {
  const l = LESIONS[code];
  return `
  <article class="lesion">
    <div class="lesion__img">
      <img src="assets/lesions/${code}.jpg" alt="Dermoscopy image of ${l.name.toLowerCase()}" loading="lazy">
      <span class="tag ${l.malignant ? "tag--m" : "tag--b"}">${l.malignant ? "malignant" : "benign"}</span>
    </div>
    <div class="lesion__body">
      <div class="lesion__head">
        <div><span class="lesion__code">${code}</span><h3>${l.name}</h3></div>
        <div class="lesion__metric"><strong data-recall="${code}">–</strong><small>recognised</small></div>
      </div>
      <p>${l.text}</p>
      <div class="lesion__foot"><span>${l.n.toLocaleString()} images</span><span>${pct(l.n / TOTAL, 1)} of data</span></div>
    </div>
  </article>`;
}).join("");
$$("[data-scroll]").forEach((btn) => btn.addEventListener("click", () => {
  const card = $(".lesion", carousel);
  carousel.scrollBy({ left: Number(btn.dataset.scroll) * (card.offsetWidth + 20), behavior: reduceMotion ? "auto" : "smooth" });
}));

/* ---------- How it works ---------- */
const stepList = $("#steps");
const steps = $$("li", stepList);
const panes = $$("#howVisual .how__pane");
let stepTimer = null;
let touched = false;
function showStep(i) {
  steps.forEach((s, k) => s.classList.toggle("is-active", k === i));
  panes.forEach((p, k) => p.classList.toggle("is-active", k === i));
}
function stopAuto() { clearInterval(stepTimer); stepTimer = null; stepList.classList.remove("is-auto"); }
steps.forEach((s, i) => $("button", s).addEventListener("click", () => { touched = true; stopAuto(); showStep(i); }));
if (!reduceMotion) {
  new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting && stepTimer === null && !touched) {
      let i = steps.findIndex((s) => s.classList.contains("is-active"));
      stepList.classList.add("is-auto");
      showStep(i);
      stepTimer = setInterval(() => { i = (i + 1) % steps.length; showStep(i); }, 4200);
    } else if (!entry.isIntersecting && stepTimer !== null) {
      stopAuto();
    }
  }, { threshold: 0.35 }).observe($("#how"));
}

/* ---------- Parallax band & reduced motion ---------- */
const bandMedia = $(".band__media");
if (!reduceMotion && bandMedia) {
  const band = bandMedia.parentElement;
  window.addEventListener("scroll", () => {
    const r = band.getBoundingClientRect();
    if (r.bottom < 0 || r.top > innerHeight) return;
    bandMedia.style.setProperty("--parallax", `${((r.top + r.height / 2 - innerHeight / 2) * -0.12).toFixed(1)}px`);
  }, { passive: true });
}
if (reduceMotion) $$("video").forEach((v) => { v.removeAttribute("autoplay"); v.pause(); });

/* ---------- Tooltip ---------- */
const tip = $("#tooltip");
function showTip(el) {
  tip.innerHTML = el.dataset.tip;
  tip.hidden = false;
  const r = el.getBoundingClientRect();
  const x = Math.min(Math.max(r.left + r.width / 2, 140), innerWidth - 140);
  tip.style.left = `${x}px`;
  tip.style.top = `${r.top}px`;
}
const hideTip = () => { tip.hidden = true; };
document.addEventListener("mouseover", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(t); });
document.addEventListener("mouseout", (e) => { if (e.target.closest("[data-tip]")) hideTip(); });
document.addEventListener("focusin", (e) => { const t = e.target.closest("[data-tip]"); if (t) showTip(t); });
document.addEventListener("focusout", hideTip);
window.addEventListener("scroll", hideTip, { passive: true });

/* ---------- Results charts ---------- */
// Sequential ramp for the confusion matrix on the dark surface: one hue family, dark -> light = low -> high.
const RAMP = [[44, 36, 32], [107, 58, 44], [168, 67, 43], [216, 115, 90], [241, 201, 184]];
function rampColor(v) {
  const x = Math.min(Math.max(v, 0), 1) * (RAMP.length - 1);
  const i = Math.min(Math.floor(x), RAMP.length - 2);
  const t = x - i;
  const c = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * t));
  return `rgb(${c.join(",")})`;
}

let modelInfo = null;
let chartsShown = false;

function buildCharts(m) {
  // KPI values
  if (m.cv) {
    $$("[data-metric]").forEach((el) => {
      const v = m.cv[el.dataset.metric];
      if (v != null) el.textContent = el.dataset.metric === "macro_auc" ? v.toFixed(3) : pct(v, 1);
    });
  }
  // Recall per class
  const per = Object.fromEntries((m.per_class || []).map((r) => [r.class, r]));
  CARD_ORDER.forEach((c) => { const r = per[c]; const el = $(`[data-recall="${c}"]`); if (r && el) el.textContent = pct(r.recall); });
  $("#recallChart").innerHTML = CARD_ORDER.filter((c) => per[c]).map((c) => {
    const r = per[c], l = LESIONS[c];
    const t = `<strong>${esc(l.name)}</strong>Recall ${pct(r.recall, 1)} · precision ${pct(r.precision, 1)}<br>AUC ${r.auc.toFixed(3)} · ${r.support.toLocaleString()} validation images`;
    return `<li class="${l.malignant ? "is-malignant" : ""}" tabindex="0" data-tip="${esc(t)}" aria-label="${esc(l.name)}: recall ${pct(r.recall, 1)}">
      <span>${esc(l.name)}</span><i data-w="${r.recall.toFixed(4)}"></i><b>${pct(r.recall)}</b></li>`;
  }).join("") + "";
  const legend = document.createElement("div");
  legend.className = "chart-legend";
  legend.innerHTML = `<span style="--c: var(--clay)">Malignant type</span><span style="--c: var(--blush)">Benign type</span>`;
  $("#recallChart").after(legend);

  // One-line insight: the weakest class and what it is most often mistaken for
  if (m.confusion && m.per_class?.length) {
    const worst = [...m.per_class].sort((a, b) => a.recall - b.recall)[0];
    const i = CLASSES.indexOf(worst.class);
    const row = m.confusion[i];
    const total = row.reduce((a, b) => a + b, 0) || 1;
    const j = row.map((n, k) => (k === i ? -1 : n)).reduce((best, n, k, arr) => (n > arr[best] ? k : best), 0);
    const box = document.createElement("p");
    box.className = "insight";
    box.innerHTML = `<span class="insight__label">Hardest case</span><span><strong>${esc(LESIONS[worst.class].name)}</strong> is recognised ${pct(worst.recall)} of the time; it is most often mistaken for <strong>${esc(LESIONS[CLASSES[j]].name.toLowerCase())}</strong> (${pct(row[j] / total)}), which can look very similar.</span>`;
    legend.after(box);
  }

  // Confusion matrix (row-normalised)
  if (m.confusion) {
    const cm = m.confusion;
    let html = `<span></span>` + CLASSES.map((c) => `<span class="cm__h" title="Predicted ${esc(LESIONS[c].name)}">${c}</span>`).join("");
    cm.forEach((row, i) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      html += `<span class="cm__h cm__h--row" title="True ${esc(LESIONS[CLASSES[i]].name)}">${CLASSES[i]}</span>`;
      row.forEach((n, j) => {
        const v = n / total;
        const label = v >= 0.01 ? Math.round(v * 100) : "";
        const t = `<strong>${esc(LESIONS[CLASSES[i]].name)}</strong>predicted as ${esc(LESIONS[CLASSES[j]].name)}: ${n.toLocaleString()} of ${total.toLocaleString()} (${pct(v, 1)})`;
        html += `<span class="cm__c${i === j ? " is-diag" : ""}" tabindex="0" data-tip="${esc(t)}" data-v="${v.toFixed(4)}"
          style="background:${rampColor(0)};color:${v > 0.55 ? "var(--ink)" : "var(--white)"}" aria-label="${esc(t.replace(/<[^>]+>/g, " "))}">${label}</span>`;
      });
    });
    $("#confusion").innerHTML = html;
  }

  // Calibration before/after
  const cal = m.calibration || {};
  if (cal.oof_before && cal.oof_after) {
    const max = Math.max(cal.oof_before.ece, cal.oof_after.ece, 0.01);
    [["before", cal.oof_before.ece], ["after", cal.oof_after.ece]].forEach(([k, v]) => {
      $(`[data-calib="${k}"]`).dataset.w = (v / max).toFixed(4);
      $(`[data-calib-val="${k}"]`).textContent = pct(v, 1);
    });
  }

  // Notes
  if (m.n_images && m.n_folds) {
    $("#resultsNote").textContent = `Pooled validation predictions on ${m.n_images.toLocaleString()} images across ${m.n_folds} cross-validation folds, split by lesion so the same mole never appears in both training and validation.`;
  }
  let foot = `Model: ${m.ensemble_size} × ${m.backbone} at ${m.img_size}px, calibrated with temperature ${m.temperature.toFixed(2)}; the live demo averages ${m.tta_views} view${m.tta_views > 1 ? "s" : ""} per image.`;
  if (m.test) foot += ` Held-out ISIC 2018 test set: balanced accuracy ${pct(m.test.balanced_acc, 1)}, macro AUC ${m.test.macro_auc.toFixed(3)}.`;
  $("#resultsFoot").textContent = foot;
}

function animateCharts() {
  if (chartsShown || !modelInfo) return;
  chartsShown = true;
  const apply = () => {
    $$("[data-meter]").forEach((el) => { const v = modelInfo.cv?.[el.dataset.meter]; if (v != null) el.style.setProperty("--v", v); });
    $$("#recallChart i, .calib__row i").forEach((el) => { if (el.dataset.w) el.style.setProperty("--w", el.dataset.w); });
    $$(".cm__c").forEach((el, k) => {
      const v = Number(el.dataset.v);
      setTimeout(() => { el.style.background = rampColor(v); }, reduceMotion ? 0 : (k % 7) * 40 + Math.floor(k / 7) * 60);
    });
  };
  requestAnimationFrame(() => requestAnimationFrame(apply));
}
new IntersectionObserver(([e]) => { if (e.isIntersecting) animateCharts(); }, { threshold: 0.15 }).observe($("#results"));

fetch("api/model").then((r) => (r.ok ? r.json() : Promise.reject(r))).then((m) => {
  modelInfo = m;
  $$("[data-model]").forEach((el) => { if (m[el.dataset.model]) el.dataset.to = m[el.dataset.model]; });
  buildCharts(m);
  const rect = $("#results").getBoundingClientRect();
  if (rect.top < innerHeight && rect.bottom > 0) animateCharts();
}).catch(() => {
  $("#resultsFoot").textContent = "Live model statistics are unavailable right now.";
});

/* ---------- Analyser ---------- */
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const preview = $("#preview");
const ui = { idle: $("#resultIdle"), loading: $("#resultLoading"), error: $("#resultError"), body: $("#resultBody") };
function showState(name) { for (const [k, el] of Object.entries(ui)) el.hidden = k !== name; }

function setPreview(file) {
  if (preview.src.startsWith("blob:")) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);
  preview.hidden = false;
  $("#dropEmpty").hidden = true;
}

async function analyze(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png|webp)$/.test(file.type)) { showError("Please choose a JPEG, PNG or WebP image."); return; }
  if (file.size > 15 * 1024 * 1024) { showError("That image is larger than 15 MB."); return; }
  setPreview(file);
  showState("loading");
  const form = new FormData();
  form.append("image", file);
  try {
    const res = await fetch("api/analyze", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `The server returned an error (${res.status}).`);
    render(data);
  } catch (err) {
    showError(err.message === "Failed to fetch" ? "Couldn't reach the server. Please try again in a moment." : err.message);
  }
}

function showError(msg) { ui.error.textContent = msg; showState("error"); }

function render(d) {
  $("#topName").textContent = d.top.name;
  $("#topKind").innerHTML = `Most likely <span class="tag ${d.top.malignant ? "tag--m" : "tag--b"}">${d.top.malignant ? "malignant type" : "benign type"}</span>`;
  $("#topProb").textContent = pct(d.top.prob);
  $("#topCode").textContent = d.top.code;
  const gauge = $("#gauge");
  gauge.style.setProperty("--p", "0");
  gauge.classList.toggle("is-low", d.malignant_prob < 0.3);
  requestAnimationFrame(() => requestAnimationFrame(() => gauge.style.setProperty("--p", d.malignant_prob.toFixed(3))));
  $("#malignantProb").textContent = pct(d.malignant_prob);
  $("#bars").innerHTML = d.probs.map((p) => `
    <li class="${p.malignant ? "is-malignant" : ""}">
      <span title="${esc(p.name)}">${esc(p.name)}</span><b>${pct(p.prob, p.prob < 0.1 ? 1 : 0)}</b><i style="--w:${p.prob.toFixed(4)}"></i>
    </li>`).join("");
  $("#inputImg").src = d.input_image;
  $("#heatImg").src = d.heatmap;
  showState("body");
  if (innerWidth < 1080) $("#result").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

fileInput.addEventListener("change", () => { $$("#samples button").forEach((b) => b.classList.remove("is-active")); analyze(fileInput.files[0]); });
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("is-over"); }));
dropzone.addEventListener("drop", (e) => analyze(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) analyze(item.getAsFile());
});

$$("#samples button").forEach((btn) => btn.addEventListener("click", async () => {
  $$("#samples button").forEach((b) => b.classList.toggle("is-active", b === btn));
  const code = btn.dataset.sample;
  try {
    const blob = await (await fetch(`assets/lesions/${code}.jpg`)).blob();
    analyze(new File([blob], `${code}.jpg`, { type: "image/jpeg" }));
  } catch { showError("Couldn't load the example image."); }
}));

const compare = $("#compare");
$("#compareRange").addEventListener("input", (e) => compare.style.setProperty("--x", `${e.target.value}%`));

$("#year").textContent = new Date().getFullYear();
