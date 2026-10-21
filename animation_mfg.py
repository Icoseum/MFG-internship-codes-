"""
Animation comparative : blind (Step 0) vs corrige (Upgrade I).

Prerequis : pip install numpy scipy matplotlib pillow
Place sim_data.mat (export MATLAB) dans le meme dossier que ce script.

Sous Spyder : pour voir l'animation en direct, passer le backend graphique
sur "Automatique" (Outils > Preferences > Console IPython > Graphique).
Le GIF est de toute facon sauvegarde sur le disque.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.io import loadmat

# ----------------------------- REGLAGES -----------------------------------
FICHIER      = "sim_data.mat"
SORTIE_GIF   = "mfg_blind_vs_corrige.gif"
N_AFFICHES   = 400      # nombre d'agents dessines (None = tous)
PAS_TEMPS    = 2        # 1 = toutes les images, 2 = une sur deux, ...
FPS          = 20
TRAINEE      = 8        # longueur de la trainee en pas de temps (0 = aucune)
SEED         = 0
COULEURS     = ["#4C78A8", "#54A24B", "#F58518", "#B279A2"]   # d1..d4
# --------------------------------------------------------------------------

d = loadmat(FICHIER)
trajs = {"blind": d["trajs_base"], "corrige": d["trajs_fin"]}      # N x T x 2
cible = {"blind": d["ad_base_idx"].ravel().astype(int) - 1,        # 0..3
         "corrige": d["ad_fin_idx"].ravel().astype(int) - 1}
tgt = d["tgt"]                                                     # 4 x 2
t_grid = d["t_grid"].ravel()

N, N_time, _ = trajs["blind"].shape
rng = np.random.default_rng(SEED)
idx = np.arange(N) if N_AFFICHES is None else rng.choice(N, min(N_AFFICHES, N), replace=False)
frames = np.arange(0, N_time, PAS_TEMPS)

# Parts de population par cible (sur TOUS les agents, pas seulement ceux affiches)
parts = {k: 100 * np.bincount(cible[k][cible[k] >= 0], minlength=4) / N for k in cible}

# Limites des axes : on inclut les trajectoires (les spirales peuvent sortir de [-10,10])
tout = np.concatenate([trajs["blind"].reshape(-1, 2), trajs["corrige"].reshape(-1, 2)])
lim = max(11.0, np.abs(tout).max() * 1.05)

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6))
titres = {"blind": "Sans correction (congestion-blind)",
          "corrige": "Avec correction de congestion (Upgrade I)"}
scat, lignes, textes = {}, {}, {}

for ax, k in zip(axes, ["blind", "corrige"]):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.set_title(titres[k], fontsize=12)
    for j in range(4):
        ax.scatter(*tgt[j], marker="*", s=380, c="yellow", edgecolors="k",
                   linewidths=1.3, zorder=5)
        ax.annotate(f"d{j+1}\n{parts[k][j]:.0f}%", tgt[j], textcoords="offset points",
                    xytext=(0, 14 if tgt[j][1] > 0 else -30), ha="center",
                    fontsize=10, fontweight="bold")
    cols = [COULEURS[c] if c >= 0 else "grey" for c in cible[k][idx]]
    scat[k] = ax.scatter(trajs[k][idx, 0, 0], trajs[k][idx, 0, 1], s=14, c=cols, zorder=4)
    lignes[k] = []
    if TRAINEE > 0:
        for i, c in zip(idx, cols):
            ln, = ax.plot([], [], lw=0.7, c=c, alpha=0.45, zorder=3)
            lignes[k].append(ln)
    textes[k] = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=11)

fig.suptitle("Trajectoires en boucle fermee : meme population initiale, "
             "seul le choix de cible differe", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])


def maj(f):
    for k in ("blind", "corrige"):
        scat[k].set_offsets(trajs[k][idx, f, :])
        if TRAINEE > 0:
            f0 = max(0, f - TRAINEE)
            for n_, ln in zip(range(len(idx)), lignes[k]):
                seg = trajs[k][idx[n_], f0:f + 1, :]
                ln.set_data(seg[:, 0], seg[:, 1])
        textes[k].set_text(f"t = {t_grid[f]:.2f} s")
    return []


anim = FuncAnimation(fig, maj, frames=frames, interval=1000 / FPS, blit=False)
anim.save(SORTIE_GIF, writer=PillowWriter(fps=FPS), dpi=110)
print(f"GIF sauvegarde : {SORTIE_GIF}")
plt.show()