import h5py
import numpy as np
import matplotlib.pyplot as plt

ASCAD_PATH = '/mnt/g/DATA/ascad/ASCAD_data/ASCAD_databases/ASCAD.h5'

with h5py.File(ASCAD_PATH, 'r') as f:
    X_prof  = f['Profiling_traces/traces'][:]
    md_prof = f['Profiling_traces/metadata'][:]
    X_att   = f['Attack_traces/traces'][:]
    md_att  = f['Attack_traces/metadata'][:]

pt_prof   = md_prof['plaintext']
key_prof  = md_prof['key']
mask_prof = md_prof['masks']
pt_att    = md_att['plaintext']
key_att   = md_att['key']

TRUE_KEY = key_att[0]
print("Shape profiling :", X_prof.shape)
print("Shape attack    :", X_att.shape)
print("Vraie cle (hex) :", ' '.join(f'{b:02X}' for b in TRUE_KEY))

# Visualisation
fig, axes = plt.subplots(2, 1, figsize=(14, 6))

axes[0].plot(X_prof[:5].T, alpha=0.6, linewidth=0.8)
axes[0].set_title("5 traces brutes ASCAD (profiling)")
axes[0].set_xlabel("Echantillon")
axes[0].set_ylabel("Amplitude")

axes[1].plot(X_prof[:500].mean(axis=0), color='navy', linewidth=1)
axes[1].set_title("Trace moyenne sur 500 traces")
axes[1].set_xlabel("Echantillon")
axes[1].set_ylabel("Amplitude moyenne")

plt.tight_layout()
plt.savefig('../results/01_traces_brutes.png', dpi=150)
print("Graphe sauvegarde : 01_traces_brutes.png")
