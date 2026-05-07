"""발표용 성능 progression 바 차트."""
import matplotlib
matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

methods = ["DOPE\n(baseline)", "CoordDOPE", "CoordDOPE\n+ Self-Training"]
values = [18.9, 21.6, 60.5]
colors = ["#B0BEC5", "#64B5F6", "#E53935"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(methods, values, color=colors, edgecolor="#263238", linewidth=1.5, width=0.55)

# 값 라벨
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
            ha="center", va="bottom", fontsize=18, fontweight="bold")

# 개선 화살표
def arrow(x0, x1, y, delta, color="#37474F"):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="->", lw=2, color=color))
    ax.text((x0 + x1) / 2, y + 1.2, f"+{delta:.1f}pp",
            ha="center", va="bottom", fontsize=13, color=color, fontweight="bold")

arrow(0, 1, 30, 21.6 - 18.9)
arrow(1, 2, 45, 60.5 - 21.6, color="#C62828")

ax.set_ylabel("Per-frame <20px 정확도 (%)", fontsize=14)
ax.set_ylim(0, 75)
ax.set_title("모델별 성능 향상 (Real test, middle 440 frames)",
             fontsize=16, fontweight="bold", pad=15)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", labelsize=12)

plt.tight_layout()
out = "_docs/figures/progression_chart.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
print(f"저장: {out}")
