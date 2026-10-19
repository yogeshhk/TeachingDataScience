"""
Charts for the "Frugal AI" seminar (LaTeX/Main_Seminar_AI_Frugal_*). Every number is one that a source states;
the source is named in the chart note and in the deck's source list.

  frugal_gemma_sizes.png        Gemma 2 2B instruct GGUF file sizes by quantisation
  frugal_energy_per_prompt.png  energy per prompt: one measured figure and one estimate, different methods
  frugal_size_vs_memory.png     a 2B model file against the RAM of an Arduino, an old laptop and a Raspberry Pi

Sources
  Gemma sizes: huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF (file sizes in GB, read 2026-10-19)
  Energy: Google, "Measuring the environmental impact of delivering AI at Google Scale" (arXiv 2508.15734),
          median text prompt 0.24 Wh within a full-stack boundary; Jegham et al., "How Hungry is AI?"
          (arXiv 2505.09598), over 29 Wh for a long prompt on the most energy-intensive of 30 models, estimated
          from public API data
  Memory: Nordic nRF52840 (256 KB RAM, as on the Arduino Nano 33 BLE Sense), Raspberry Pi 5 (1 to 16 GB RAM
          options), AIOOLL README (an 8 GB laptop). Decimal units throughout.

Run: conda activate genai && python make_charts.py   (writes into ../../LaTeX/images)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "..", "LaTeX", "images"))
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, GREY, GREEN = "#2b6cb0", "#dd6b20", "#a0aec0", "#2f855a"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---- 1. Gemma 2 2B file sizes -------------------------------------------------------------------
sizes = [("IQ3_M", 1.39), ("Q3_K_L", 1.55), ("IQ4_XS", 1.57), ("Q4_K_M", 1.71), ("Q5_K_M", 1.92),
         ("Q6_K", 2.15), ("Q8_0", 2.78), ("F32 (full precision)", 10.5)]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
names = [n for n, _ in sizes][::-1]
vals = [v for _, v in sizes][::-1]
colors = [ORANGE if n == "Q4_K_M" else (GREY if n.startswith("F32") else BLUE) for n in names]
bars = ax.barh(names, vals, color=colors)
for bar, v in zip(bars, vals):
    ax.text(v + 0.12, bar.get_y() + bar.get_height() / 2, "%.2f GB" % v, va="center", fontsize=10)
ax.set_xlim(0, 12.5)
ax.set_xlabel("File size (GB)")
ax.set_title("Gemma 2 2B instruct: the same model, fewer bits per weight", fontsize=11)
save(fig, "frugal_gemma_sizes.png")   # source is named on the slide

# ---- 2. Energy per prompt -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 3.0))
labels = ["Median text prompt\n(Google, measured,\nfull-stack boundary)",
          "Long prompt, most energy-\nintensive of 30 models\n(benchmark, estimated)"]
vals = [0.24, 29.0]
bars = ax.bar(labels, vals, color=[BLUE, ORANGE], width=0.5)
ax.set_yscale("log")
ax.set_ylim(0.1, 80)
ax.set_ylabel("Energy per prompt (Wh, log scale)")
ax.text(0, 0.24 * 1.25, "0.24 Wh", ha="center", fontsize=11)
ax.text(1, 29 * 1.2, "over 29 Wh", ha="center", fontsize=11)
ax.set_title("Two published figures: different methods, so not a like-for-like comparison", fontsize=10)
save(fig, "frugal_energy_per_prompt.png")   # sources are named on the slide

# ---- 3. Model file against device memory -------------------------------------------------------
items = [("Arduino Nano 33 BLE Sense: RAM", 256e3, GREEN),
         ("Gemma 2 2B Q4_K_M: file size", 1.71e9, ORANGE),
         ("Raspberry Pi 5: smallest RAM option", 1e9, BLUE),
         ("Old laptop in AIOOLL: RAM", 8e9, BLUE),
         ("Raspberry Pi 5: largest RAM option", 16e9, BLUE)]
items.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.barh([n for n, _, _ in items], [v for _, v, _ in items], color=[c for _, _, c in items])
ax.set_xscale("log")
ax.set_xlim(1e5, 8e10)
ticks = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
ax.set_xticks(ticks)
ax.set_xticklabels(["100 KB", "1 MB", "10 MB", "100 MB", "1 GB", "10 GB"])
for n, v, _ in items:
    label = ("%d KB" % round(v / 1e3)) if v < 1e6 else ("%.2f GB" % (v / 1e9)).rstrip("0").rstrip(".")
    ax.text(v * 1.15, [i for i, (m, _, _) in enumerate(items) if m == n][0], label, va="center", fontsize=10)
ax.set_title("A 2B-parameter model file is about 6,700 times the RAM of this microcontroller", fontsize=10)
fig.text(0.01, -0.05, "Sources: Nordic nRF52840, Raspberry Pi 5 page, AIOOLL README, Hugging Face file listing. Decimal units.",
         fontsize=8, ha="left")
save(fig, "frugal_size_vs_memory.png")

ratio = 1.71e9 / 256e3
print("Gemma Q4_K_M file / Arduino RAM = %.0f (decimal units); F32 / Q4_K_M = %.2f" % (ratio, 10.5 / 1.71))
