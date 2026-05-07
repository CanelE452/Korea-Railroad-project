"""F4/F5 epoch sweep learning curve plot."""
import matplotlib.pyplot as plt
import numpy as np

epochs_31 = [65, 70, 75, 80, 85, 90, 95, 96]
epochs_60 = [65, 70, 75, 80, 85, 90, 95, 96, 100, 105, 110, 115, 120, 125]

f5_31 = [21.6, 20.9, 39.8, 47.7, 55.5, 58.2, 60.0, 60.5]
f5_60 = [21.6, 20.9, 39.8, 47.7, 55.5, 58.2, 60.0, 60.5, None, None, None, None, None, 59.5]

fig, ax = plt.subplots(figsize=(10, 5.5))

# F5 full (60 epoch)
ep_plot = []
val_plot = []
for e, v in zip(epochs_60, f5_60):
    if v is not None:
        ep_plot.append(e)
        val_plot.append(v)
ax.plot(ep_plot, val_plot, 'o-', color='#e74c3c', linewidth=2.5, markersize=7, label='F5: RANSAC+LOO (2 PL)')

# Best marker
ax.plot(96, 60.5, '*', color='#e74c3c', markersize=20, zorder=5)
ax.annotate('★ best: 60.5%\n(ep96, 31 epochs ft)', xy=(96, 60.5), xytext=(103, 55),
            fontsize=11, fontweight='bold', color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5))

# ep125
ax.annotate('ep125: 59.5%\n(60 epochs ft)', xy=(125, 59.5), xytext=(113, 48),
            fontsize=10, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5))

# Baseline
ax.axhline(y=21.6, color='gray', linestyle=':', linewidth=1, alpha=0.7)
ax.text(66, 23.0, 'CoordDOPE baseline (21.6%)', fontsize=9, color='gray')

# Fine-tune epochs annotation
ax.annotate('', xy=(65, 2), xytext=(96, 2),
            arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.5))
ax.text(78, 3.5, '31 epochs', fontsize=10, ha='center', color='#2c3e50', fontweight='bold')

ax.annotate('', xy=(65, 6), xytext=(125, 6),
            arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5))
ax.text(95, 7.5, '60 epochs', fontsize=10, ha='center', color='#7f8c8d')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('NN matching <20px (%)', fontsize=12)
ax.set_title('Self-Training: Best Epoch at 31 (ep96)', fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim(0, 70)
ax.set_xlim(62, 130)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = '_docs/figures/f4_f5_epoch_sweep.png'
plt.savefig(out, dpi=150)
print(f'저장: {out}')
