from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

OUT = Path("assets/figures/pa_soft_knn.png")
W, H = 15.6, 8.2
fig, ax = plt.subplots(figsize=(W, H), dpi=160)
ax.set_xlim(0, 15.6)
ax.set_ylim(0, 8.2)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ink = "#202826"
teal = "#287c80"
line_teal = "#9ed2cf"
muted = "#5d6a67"
paper = "#fffdf8"
line = "#d8cfbf"
colors = {
    "c1": "#5fa276",
    "c2": "#e5a844",
    "c3": "#9065bc",
    "c4": "#cf6669",
}

# Heatmap field: local softmax mass around top-k candidates. Low tau leaves sharp peaks;
# higher tau spreads probability mass to nearby candidates.
x = np.linspace(3.0, 10.4, 360)
y = np.linspace(1.45, 6.7, 240)
X, Y = np.meshgrid(x, y)
cands = [
    (4.65, 5.55, 0.92, "c1"),
    (5.95, 3.05, 0.84, "c1"),
    (7.25, 4.25, 0.75, "c2"),
    (8.62, 2.05, 0.69, "c3"),
    (9.85, 5.40, 0.61, "c4"),
]
tau = 0.18
scores = np.array([c[2] for c in cands])
weights = np.exp(scores / tau)
weights = weights / weights.sum()
Z = np.zeros_like(X)
for (cx, cy, score, _), wt in zip(cands, weights):
    sigma = 0.36 + 0.95 * tau
    Z += wt * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))
Z = (Z / Z.max()) ** 0.55
cmap = LinearSegmentedColormap.from_list("pa_heat", ["#ffffff", "#d8f1d5", "#8acb83", "#228b22"])
ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", cmap=cmap, alpha=0.62, interpolation="bilinear", zorder=0)

# Titles
ax.text(6.8, 7.55, "correspondence region similarity", ha="center", va="center", fontsize=24, fontweight="bold", color=ink)
ax.text(12.4, 7.30, "palette vote", ha="center", va="center", fontsize=24, fontweight="bold", color=ink)

# Target region
ax.text(1.25, 6.30, "target\nregion", ha="left", va="top", fontsize=22, fontweight="bold", color=ink, linespacing=1.08)
target = Circle((1.85, 3.85), 0.70, facecolor="#f3d5b5", edgecolor=ink, linewidth=2.2, zorder=4)
ax.add_patch(target)

# Top-k bracket and temperature label
ax.plot([3.55, 10.35], [6.85, 6.85], color=teal, lw=2.0)
ax.plot([3.55, 3.55], [6.67, 6.85], color=teal, lw=2.0)
ax.plot([10.35, 10.35], [6.67, 6.85], color=teal, lw=2.0)
ax.text(6.95, 6.95, "retain top-k candidates", ha="center", va="bottom", fontsize=16, fontweight="bold", color=teal)

# Similarity arrows and candidate nodes
for cx, cy, score, label in cands:
    arrow = FancyArrowPatch((2.45, 3.85), (cx - 0.68, cy - 0.02), arrowstyle="->", mutation_scale=18, lw=2.0, color=line_teal, alpha=0.70, zorder=1)
    ax.add_patch(arrow)

for cx, cy, score, label in cands:
    radius = 0.62 if label != "c4" else 0.52
    ax.add_patch(Circle((cx, cy), radius + 0.045, facecolor="white", edgecolor="#6f7977", linewidth=1.2, zorder=3))
    ax.add_patch(Circle((cx, cy), radius, facecolor=colors[label], edgecolor="#6f7977", linewidth=1.1, zorder=4))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=19, fontweight="bold", color="white", zorder=5)
    label_y = cy - radius - 0.36
    label_x = cx
    if label == "c3":
        label_x = cx + 0.22
        label_y = cy - radius - 0.20
    ax.text(label_x, label_y, f"s={score:.2f}", ha="center", va="center", fontsize=13, color=muted,
            bbox=dict(boxstyle="round,pad=0.10", facecolor="#fffdfa", edgecolor="#eadfce", alpha=0.95), zorder=6)

# Arrow to palette vote panel
ax.add_patch(FancyArrowPatch((10.25, 3.85), (11.05, 3.85), arrowstyle="simple", mutation_scale=34, lw=1.5, facecolor="#91d5d1", edgecolor="#4ba7a9", alpha=0.95, zorder=4))

# Palette vote panel
panel = FancyBboxPatch((11.25, 1.35), 3.45, 5.05, boxstyle="round,pad=0.16,rounding_size=0.15", facecolor=paper, edgecolor=line, linewidth=2.0)
ax.add_patch(panel)
rows = [("c1", 0.63), ("c2", 0.18), ("c3", 0.12), ("c4", 0.07)]
for i, (lab, val) in enumerate(rows):
    y0 = 5.35 - i * 0.95
    ax.text(11.62, y0, lab, ha="left", va="center", fontsize=16, color=ink)
    ax.add_patch(Rectangle((12.15, y0 - 0.20), 1.32, 0.40, facecolor="#ece6d8", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((12.15, y0 - 0.20), 1.32 * val / 0.63, 0.40, facecolor=colors[lab], edgecolor="none", zorder=3))
    ax.text(13.62, y0, f"{val:.2f}", ha="left", va="center", fontsize=15, color=muted)

# Colour probability from the paper, expanded to expose the temperature term.
formula = (
    r"$P_{t,i}(c)=\sum_{(r,j)\in\mathcal{N}_k(t,i)}"
    r"\left[\exp(S_{(t,i),(r,j)}/\tau)\,/\,"
    r"\sum_{(r',j')\in\mathcal{N}_k(t,i)}\exp(S_{(t,i),(r',j')}/\tau)\right]"
    r"\,\mathbf{1}\!\left[y^{(r)}_j=c\right]$"
)
ax.text(7.82, 0.62, formula, ha="center", va="center", fontsize=14.6, color=ink)

plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=160, bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
