"""Hjælpefunktioner til Specialiserede Modeller — KUN plotte-hjælp, ingen magi.

I behøver ikke læse koden her for at følge med. Men kig endelig: det er
helt almindelig matplotlib, som I kender — bare med lidt ekstra pynt.
Alt det, der er selve læringsmålet (modeller, encoding og træningsloops),
står synligt i notebooks'ene.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def _til_numpy(x):
    """Konverterer en tensor til numpy — og lader numpy-arrays være i fred."""
    if torch.is_tensor(x):
        return x.detach().numpy()
    return np.asarray(x)


def _som_2d_billeder(billeder):
    """Accepterer (N, H*W), (N, H, W) eller (N, 1, H, W) og giver (N, H, W)."""
    billeder = _til_numpy(billeder)
    if billeder.ndim == 4:                       # (N, 1, H, W)
        billeder = billeder[:, 0]
    if billeder.ndim == 2:                       # (N, H*W) — antag kvadratisk
        side = int(np.sqrt(billeder.shape[1]))
        billeder = billeder.reshape(-1, side, side)
    return billeder


def vis_billeder(billeder, labels=None, forudsigelser=None, n=10, klassenavne=None):
    """Viser n billeder i ét grid.

    billeder:       (N, H*W), (N, H, W) eller (N, 1, H, W) — tensor eller array
    labels:         (N,) de rigtige klasser (valgfri)
    forudsigelser:  (N,) modellens gæt (valgfri) — titlen bliver RØD ved fejlgæt
    klassenavne:    liste der oversætter klassenumre til navne (fx dansk tøj)
    """
    billeder = _som_2d_billeder(billeder)
    if labels is not None:
        labels = _til_numpy(labels).ravel()
    if forudsigelser is not None:
        forudsigelser = _til_numpy(forudsigelser).ravel()

    def navn(k):
        k = int(k)
        return klassenavne[k] if klassenavne is not None else str(k)

    n = min(n, len(billeder))
    kolonner = min(n, 5)
    raekker = int(np.ceil(n / kolonner))
    fig, akser = plt.subplots(raekker, kolonner,
                              figsize=(2.2 * kolonner, 2.6 * raekker))
    for i, akse in enumerate(np.atleast_1d(akser).ravel()):
        akse.axis("off")
        if i >= n:
            continue
        akse.imshow(billeder[i], cmap="gray")
        if forudsigelser is not None and labels is not None:
            rigtigt = int(forudsigelser[i]) == int(labels[i])
            akse.set_title(f"gæt: {navn(forudsigelser[i])}\n(label: {navn(labels[i])})",
                           color="black" if rigtigt else "red", fontsize=9)
        elif labels is not None:
            akse.set_title(navn(labels[i]), fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_forvirringsmatrix(y_sand, y_gaet, klassenavne=None, titel="Forvirringsmatrix"):
    """Tegner en forvirringsmatrix: rækker = det rigtige svar, kolonner = modellens gæt.

    Diagonalen er de korrekte gæt — alt udenfor er forvekslinger.
    """
    y_sand = _til_numpy(y_sand).ravel().astype(int)
    y_gaet = _til_numpy(y_gaet).ravel().astype(int)
    antal_klasser = int(max(y_sand.max(), y_gaet.max())) + 1

    matrix = np.zeros((antal_klasser, antal_klasser), dtype=int)
    for sand, gaet in zip(y_sand, y_gaet):
        matrix[sand, gaet] += 1

    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="antal")
    navne = klassenavne if klassenavne is not None else [str(i) for i in range(antal_klasser)]
    plt.xticks(range(antal_klasser), navne, rotation=45, ha="right")
    plt.yticks(range(antal_klasser), navne)
    plt.xlabel("modellens gæt")
    plt.ylabel("rigtigt svar")
    plt.title(titel)
    # skriv tallene i cellerne (hvid tekst på mørk baggrund, sort på lys)
    graense = matrix.max() / 2 if matrix.max() > 0 else 0.5
    for i in range(antal_klasser):
        for j in range(antal_klasser):
            plt.text(j, i, matrix[i, j], ha="center", va="center", fontsize=8,
                     color="white" if matrix[i, j] > graense else "black")
    plt.tight_layout()
    plt.show()


def plot_kmeans_skridt(X, centre_historik, tildelinger_historik):
    """Tegner k-means-algoritmens iterationer som et grid af scatterplots.

    X:                    (N, 2) datapunkter
    centre_historik:      liste af (k, 2)-arrays — centrene efter hver iteration
    tildelinger_historik: liste af (N,)-arrays — hvert punkts klynge-nummer pr. iteration
    """
    X = _til_numpy(X)
    antal = len(centre_historik)
    kolonner = min(antal, 3)
    raekker = int(np.ceil(antal / kolonner))
    fig, akser = plt.subplots(raekker, kolonner,
                              figsize=(4.5 * kolonner, 4 * raekker))
    for i, akse in enumerate(np.atleast_1d(akser).ravel()):
        if i >= antal:
            akse.axis("off")
            continue
        centre = _til_numpy(centre_historik[i])
        tildeling = _til_numpy(tildelinger_historik[i])
        akse.scatter(X[:, 0], X[:, 1], c=tildeling, cmap="tab10", s=15, alpha=0.7)
        akse.scatter(centre[:, 0], centre[:, 1], marker="X", s=250,
                     color="black", edgecolors="white", linewidths=1.5, zorder=3)
        akse.set_title(f"iteration {i}")
    plt.tight_layout()
    plt.show()


def vis_rekonstruktioner(originaler, rekonstruktioner, n=8, titel=""):
    """Viser originale billeder (øverste række) og rekonstruktioner (nederste)."""
    originaler = _som_2d_billeder(originaler)
    rekonstruktioner = _som_2d_billeder(rekonstruktioner)
    n = min(n, len(originaler))

    fig, akser = plt.subplots(2, n, figsize=(1.6 * n, 3.6))
    akser = np.array(akser).reshape(2, n)     # også når n = 1
    for i in range(n):
        akser[0, i].imshow(originaler[i], cmap="gray")
        akser[1, i].imshow(rekonstruktioner[i], cmap="gray")
        akser[0, i].axis("off")
        akser[1, i].axis("off")
    akser[0, 0].set_title("original", loc="left", fontsize=10)
    akser[1, 0].set_title("rekonstruktion", loc="left", fontsize=10)
    if titel:
        fig.suptitle(titel)
    plt.tight_layout()
    plt.show()
