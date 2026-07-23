# Intro til Machine Learning

Dette er materialet til emnet **Intro til Machine Learning** på KIC26 — ca. 8 timers
undervisning, der dækker PyTorch-tensorer og gradient descent, neurale netværk,
aktiveringsfunktioner og træning af hele netværk på både billeder (MNIST) og Pokémon-data.
Emnet bygger direkte oven på [Introduktion til programmering](../Intro-Programmering/)
(variabler, if-sætninger, for-løkker, funktioner, numpy, pandas, matplotlib og klasser) og
på regressionsemnet (lineær regression uden PyTorch).

## Notebooks (kør dem i rækkefølge)

| # | Notebook | Indhold | ca. tid |
|---|---|---|---|
| 1 | `1-Tensorer-autograd-og-gradient-descent.ipynb` | tensorer, autograd og gradient descent — motoren bag deep learning | 2 t |
| 2 | `2-Neurale-netvaerk.ipynb` | neuroner, `nn.Module`-klasser og det fulde træningsloop | 1,5 t |
| 3 | `3-Aktiveringsfunktioner.ipynb` | Sigmoid, ReLU, Leaky ReLU, Tanh & Softmax — og hvorfor de er nødvendige | 1,5 t |
| 4 | `4-MNIST-og-Pokemon.ipynb` | træn hele netværk: håndskrevne cifre (MNIST) OG Pokémon-typer | 3 t |
| — | `5-Ekstra-opgaver.ipynb` | store kombinerede opgaver (E.1–E.8) til de hurtige | overskud |

Til hver notebook findes en `*_løsningsforslag.ipynb` med udfyldte svar og lærernoter
(forventede accuracy-tal m.m.).

**Vigtigt til både elever og undervisere:** Der er med vilje omkring dobbelt så mange
opgaver, som tiden tillader — ingen forventes at nå alt! Opgaver mærket **(ekstra)** er
til dem, der er foran, og opgaver mærket **(find fejlen)** indeholder *bevidste* fejl, som
skal findes og rettes (så bliv ikke forskrækket, hvis "Run all" fejler dér).

## Sådan åbnes materialet (Google Colab)

Åbn en notebook direkte fra GitHub:

```
https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Intro-ML/<notebook-navn>.ipynb
```

fx [1-Tensorer-autograd-og-gradient-descent.ipynb](https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Intro-ML/1-Tensorer-autograd-og-gradient-descent.ipynb).

Alle notebooks kører fint på CPU — der skal **ikke** vælges GPU-runtime.

## Data & Plan B

Alle data og `hjaelpefunktioner.py` hentes med en `!wget`-celle fra GitHub, så der ikke
kræves Kaggle-login eller `pip install` på Colab. Filerne ligger i `data/`:

- `Pokemon.csv` — Pokémon-stats (brugt i notebook 1, 2 og 4)
- `mnist_traen_lille.csv.gz` / `mnist_test_lille.csv.gz` — nedskaleret MNIST (notebook 4)

**Plan B:** Fejler `wget`-cellen (dårligt camp-wifi), så upload de nævnte filer manuelt
via Colabs filpanel (mappeikonet til venstre) — de ligger i `Intro-ML/data` på GitHub.
`hjaelpefunktioner.py` (fire små plottefunktioner — al rigtig kode står i notebooks'ene)
hentes på samme måde. Den eneste undtagelse er opgave **E.8** i ekstra-notebooken, som
bevidst bruger `kagglehub` til at hente elevernes *eget* valgte Kaggle-datasæt.

## Vedligehold

MNIST-filerne i `data/` er stratificerede subsamples (16000 trænings- og 2000 testrækker,
lige mange pr. ciffer, seed 42) af `oddrationale/mnist-in-csv`, genereret med:

```python
import kagglehub, pandas as pd
sti = kagglehub.dataset_download("oddrationale/mnist-in-csv")
for fil, n, ud in [("mnist_train.csv", 16000, "mnist_traen_lille.csv.gz"),
                   ("mnist_test.csv", 2000, "mnist_test_lille.csv.gz")]:
    hel = pd.read_csv(f"{sti}/{fil}")
    dele = [hel[hel["label"] == c].sample(n=n // 10, random_state=42) for c in range(10)]
    pd.concat(dele).sample(frac=1, random_state=42).reset_index(drop=True).to_csv(
        f"data/{ud}", index=False, compression="gzip")
```

Den tidligere `1-Data-manipulation.ipynb` (pandas/datarensning) er flyttet til
[Introduktion til programmering](../Intro-Programmering/), da eleverne nu møder pandas dér.

Notebooks committes med ryddede outputs (ellers vokser repoet vildt).
