import os
import matplotlib.pyplot as plt

# 1. Define your data (extracted from your 30-epoch logs)
epochs = list(range(1, 31))

train_loss = [
    0.9601, 0.5382, 0.3710, 0.3000, 0.2447, 0.2253, 0.2035, 0.2128, 0.1801, 0.1709,
    0.1840, 0.1760, 0.1630, 0.1500, 0.1466, 0.1571, 0.1436, 0.1400, 0.1297, 0.1285,
    0.1252, 0.1256, 0.1190, 0.1174, 0.1157, 0.1153, 0.1151, 0.1127, 0.1116, 0.1127
]

val_loss = [
    0.9848, 0.4668, 0.3512, 0.1894, 0.3568, 0.1860, 0.2652, 0.2385, 0.2064, 0.2127,
    0.1755, 0.3031, 0.1887, 0.2159, 0.2655, 0.1935, 0.2213, 0.1492, 0.2074, 0.2252,
    0.1829, 0.1620, 0.1714, 0.1505, 0.1162, 0.1379, 0.1860, 0.1400, 0.1997, 0.1509
]

val_f1 = [
    0.4779, 0.7442, 0.7760, 0.8677, 0.7543, 0.8646, 0.8281, 0.8260, 0.8556, 0.8460,
    0.8768, 0.7872, 0.8781, 0.8446, 0.8217, 0.8749, 0.8613, 0.8822, 0.8534, 0.8683,
    0.8707, 0.8799, 0.8842, 0.8902, 0.9073, 0.8999, 0.8857, 0.8917, 0.8607, 0.8848
]

# 2. Build the side-by-side plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# (a) Loss Plot
ax1.plot(epochs, train_loss, label='Training loss', color='#1f77b4', linewidth=1.8)
ax1.plot(epochs, val_loss, label='Validation loss', color='#d62728', linewidth=1.8)
ax1.set_title('(a) BCE + Dice Loss', fontsize=11, loc='left')
ax1.set_xlabel('Epoch', fontsize=10)
ax1.set_ylabel('Loss', fontsize=10)
ax1.set_ylim(0.0, 1.0)
ax1.legend(frameon=True, facecolor='white')
ax1.grid(True, linestyle='--', alpha=0.5)

# (b) Validation F1 Metric Plot
ax2.plot(epochs, val_f1, label='Validation F1', color='#1f77b4', linewidth=1.8)
ax2.set_title('(b) Validation Change-Class Metrics', fontsize=11, loc='left')
ax2.set_xlabel('Epoch', fontsize=10)
ax2.set_ylabel('Score', fontsize=10)
ax2.set_ylim(0.4, 1.0)

# Annotate best epoch
best_ep, best_f1_val = 25, val_f1[24]
ax2.scatter(best_ep, best_f1_val, color='#1f77b4', s=40, zorder=5)
ax2.annotate(f'best F1 {best_f1_val:.4f} (epoch {best_ep})', 
             xy=(best_ep, best_f1_val), 
             xytext=(best_ep - 12, best_f1_val - 0.12),
             arrowprops=dict(arrowstyle='->', lw=0.8, color='black'),
             fontsize=9)

ax2.legend(frameon=True, facecolor='white')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# 3. Specify repository save directory (e.g., assets/ or figures/)
output_dir = './figures'
os.makedirs(output_dir, exist_ok=True)

# Save high-res PNG and JPEG into repository directory
plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'training_metrics.jpg'), dpi=300, bbox_inches='tight', pil_kwargs={'quality': 95})

plt.close()
print("Images successfully saved to repository folder: ./figures/")