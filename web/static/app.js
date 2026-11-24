"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const pct = (p, digits = 0) => `${(p * 100).toFixed(digits)}%`;

// Counts are from the HAM10000 metadata (10,015 images).
const LESIONS = [
  { code: "mel", name: "Melanoma", n: 1113, malignant: true,
    text: "Cancer of the pigment cells. Often asymmetric, with several colours and an irregular network. The most dangerous type to miss." },
  { code: "nv", name: "Melanocytic nevus", n: 6705, malignant: false,
    text: "The common mole. Usually symmetric, with an even pigment network. Two-thirds of the dataset." },
  { code: "bcc", name: "Basal cell carcinoma", n: 514, malignant: true,
    text: "The most common skin cancer. Grows slowly and rarely spreads. Branching vessels and blue-grey nests are typical." },
  { code: "akiec", name: "Actinic keratosis", n: 327, malignant: true,
    text: "Sun-damage lesions and intraepithelial carcinoma (Bowen's disease), which can progress to squamous cell carcinoma." },
  { code: "bkl", name: "Benign keratosis", n: 1099, malignant: false,
    text: "Seborrhoeic keratoses, solar lentigines and lichen-planus-like keratoses: harmless, but they can imitate melanoma." },
  { code: "df", name: "Dermatofibroma", n: 115, malignant: false,
    text: "A firm, benign nodule. Often a white central patch ringed by a fine pigment network." },
  { code: "vasc", name: "Vascular lesion", n: 142, malignant: false,
    text: "Cherry angiomas, angiokeratomas and pyogenic granulomas, with red to purple blood-filled lacunae." },
];

/* ---------- Nav & menu ---------- */
const nav = $("#nav");
const menu = $("#menu");
const toggle = $("#menuToggle");
const onScroll = () => nav.classList.toggle("is-solid", window.scrollY > 40 || !menu.hidden);
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

function setMenu(open) {
  menu.hidden = !open;
  toggle.setAttribute("aria-expanded", String(open));
  toggle.textContent = open ? "Close" : "Menu";
  document.body.style.overflow = open ? "hidden" : "";
  onScroll();
}
toggle.addEventListener("click", () => setMenu(menu.hidden));
$$("#menu a").forEach((a) => a.addEventListener("click", () => setMenu(false)));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) setMenu(false); });

/* ---------- Reveal on scroll & counters ---------- */
function animateCount(el) {
  const to = Number(el.dataset.to);
  if (reduceMotion) { el.textContent = to.toLocaleString(); return; }
  const start = performance.now();
  const dur = 1600;
  const tick = (now) => {
    const t = Math.min(1, (now - start) / dur);
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
}, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });
$$(".reveal").forEach((el) => io.observe(el));

/* ---------- Lesion carousel ---------- */
const carousel = $("#lesionCarousel");
const total = LESIONS.reduce((s, l) => s + l.n, 0);
carousel.innerHTML = LESIONS.map((l) => `
  <article class="lesion">
    <div class="lesion__img">
      <img src="assets/lesions/${l.code}.jpg" alt="Dermoscopy image of ${l.name.toLowerCase()}" loading="lazy">
      <span class="lesion__share">${pct(l.n / total, 1)} of images</span>
    </div>
    <div class="lesion__body">
      <span class="lesion__code">${l.code}<span class="tag ${l.malignant ? "tag--m" : "tag--b"}">${l.malignant ? "malignant" : "benign"}</span></span>
      <h3>${l.name}</h3>
      <p>${l.text}</p>
    </div>
  </article>`).join("");
$$("[data-scroll]").forEach((btn) => btn.addEventListener("click", () => {
  const card = $(".lesion", carousel);
  carousel.scrollBy({ left: Number(btn.dataset.scroll) * (card.offsetWidth + 20), behavior: reduceMotion ? "auto" : "smooth" });
}));

/* ---------- How it works ---------- */
const steps = $$("#steps li");
const panes = $$("#howVisual .how__pane");
let stepTimer = null;
function showStep(i) {
  steps.forEach((s, k) => s.classList.toggle("is-active", k === i));
  panes.forEach((p, k) => p.classList.toggle("is-active", k === i));
}
steps.forEach((s, i) => $("button", s).addEventListener("click", () => { clearInterval(stepTimer); stepTimer = null; showStep(i); }));
if (!reduceMotion) {
  const howObserver = new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting && stepTimer === null && !steps.some((s) => s.dataset.touched)) {
      let i = 0;
      stepTimer = setInterval(() => { i = (i + 1) % steps.length; showStep(i); }, 4200);
    } else if (!entry.isIntersecting && stepTimer !== null) {
      clearInterval(stepTimer); stepTimer = null;
    }
  }, { threshold: 0.4 });
  howObserver.observe($("#how"));
  steps.forEach((s) => s.addEventListener("click", () => { s.dataset.touched = "1"; }));
}

/* ---------- Parallax band ---------- */
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

/* ---------- Model stats ---------- */
fetch("api/model").then((r) => (r.ok ? r.json() : Promise.reject(r))).then((m) => {
  $$("[data-model]").forEach((el) => { if (m[el.dataset.model]) el.dataset.to = m[el.dataset.model]; });
  if (m.cv) {
    $$("[data-metric]").forEach((el) => {
      const v = m.cv[el.dataset.metric];
      if (v == null) return;
      el.textContent = el.dataset.metric === "macro_auc" ? v.toFixed(3) : pct(v, 1);
    });
  }
  const parts = [`${m.ensemble_size} × ${m.backbone} at ${m.img_size}px`, `${m.tta_views}-view test-time augmentation`,
                 `temperature ${m.temperature.toFixed(2)}`];
  let foot = `Model: ${parts.join(" · ")}.`;
  if (m.test) foot += ` On the held-out ISIC 2018 test set: balanced accuracy ${pct(m.test.balanced_acc, 1)}, macro AUC ${m.test.macro_auc.toFixed(3)}.`;
  $("#perfFoot").textContent = foot;
}).catch(() => {
  $("#perfFoot").textContent = "Live model statistics are unavailable right now.";
});

/* ---------- Analyser ---------- */
const fileInput = $("#fileInput");
const dropzone = $("#dropzone");
const preview = $("#preview");
const ui = {
  idle: $("#resultIdle"), loading: $("#resultLoading"), error: $("#resultError"), body: $("#resultBody"),
};
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
    showError(err.message === "Failed to fetch" ? "Couldn't reach the server. Is it running?" : err.message);
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
      <span title="${p.name}">${p.name}</span><i style="--w:${p.prob.toFixed(4)}"></i><b>${pct(p.prob, p.prob < 0.1 ? 1 : 0)}</b>
    </li>`).join("");
  $("#inputImg").src = d.input_image;
  $("#heatImg").src = d.heatmap;
  showState("body");
  if (innerWidth < 1020) $("#result").scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
}

fileInput.addEventListener("change", () => analyze(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("is-over"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("is-over"); }));
dropzone.addEventListener("drop", (e) => analyze(e.dataTransfer.files[0]));
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) analyze(item.getAsFile());
});

$$("#samples button").forEach((btn) => btn.addEventListener("click", async () => {
  const code = btn.dataset.sample;
  try {
    const blob = await (await fetch(`assets/lesions/${code}.jpg`)).blob();
    analyze(new File([blob], `${code}.jpg`, { type: "image/jpeg" }));
  } catch { showError("Couldn't load the example image."); }
}));

const compare = $("#compare");
$("#compareRange").addEventListener("input", (e) => compare.style.setProperty("--x", `${e.target.value}%`));

$("#year").textContent = new Date().getFullYear();
