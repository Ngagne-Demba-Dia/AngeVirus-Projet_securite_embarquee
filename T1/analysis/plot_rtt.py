import matplotlib.pyplot as plt
import numpy as np

positions = list(range(16))
secret    = list("S3cr3tK3y_2024!!")
rtt       = [3104.8, 3296.6, 3461.3, 3668.4, 3936.5,
             4143.7, 4313.6, 4487.9, 4697.4, 4948.7,
             5139.1, 5317.8, 5497.9, 5702.4, 5893.1, 6157.5]

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(positions, rtt, marker='o', color='steelblue', linewidth=2, markersize=8)

for i, (r, c) in enumerate(zip(rtt, secret)):
    ax.annotate(f"'{c}'", (i, r), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10, color='darkred', fontweight='bold')

ax.set_xticks(positions)
ax.set_xticklabels([f"[{i:02d}]" for i in positions])
ax.set_xlabel("Position de l'octet", fontsize=12)
ax.set_ylabel("RTT median (µs)", fontsize=12)
ax.set_title("T1 — Timing Attack sur ESP32\nRTT median par octet correct (+200 µs/octet)", fontsize=13)
ax.grid(True, alpha=0.3)

delta = np.diff(rtt)
ax.text(0.98, 0.05,
        f"Delta moyen : {delta.mean():.1f} µs/octet\nSecret : S3cr3tK3y_2024!!",
        transform=ax.transAxes, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
        fontsize=10)

plt.tight_layout()
plt.savefig("../results/03_rtt_graph.png", dpi=150)
print("Graphe sauvegarde : T1/results/03_rtt_graph.png")
plt.show()
