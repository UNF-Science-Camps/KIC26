# Intro til Machine Learning

Dette er materialet til emnet **Intro til Machine Learning** på KIC26 — ca. 9 timers
undervisning der dækker datamanipulation, PyTorch, neurale netværk og
aktiveringsfunktioner. Emnet bygger direkte oven på *Introduktion til programmering*
(variabler, if-sætninger, for-løkker, funktioner, numpy, matplotlib og klasser) og
på regressionsemnet (lineær regression uden PyTorch).

## Notebooks (kør dem i rækkefølge)

| # | Notebook | Indhold | ca. tid |
|---|---|---|---|
| 1 | `1-Data-manipulation.ipynb` | pandas, datarensning, plots og standardisering — på Pokémon-data | 2,5 t |
| 2 | `2-PyTorch-og-gradient-descent.ipynb` | tensorer, autograd, gradient descent og lineær regression i PyTorch | 2 t |
| 3 | `3-Neurale-netvaerk.ipynb` | neuroner, `nn.Module`-klasser og det fulde træningsloop | 2 t |
| 4 | `4-Aktiveringsfunktioner.ipynb` | Sigmoid, ReLU, Leaky ReLU, Tanh & Softmax — og hvorfor de er nødvendige | 1,5 t |
| 5 | `5-MNIST.ipynb` | håndskrevne cifre: alt fra emnet samlet i én model | 1 t |
| 6 | `6-Ekstra-opgaver.ipynb` | store kombinerede opgaver (E.1–E.8) til de hurtige | overskud |

Til hver notebook findes en `*_løsningsforslag.ipynb` med udfyldte svar og lærernoter
(forventede accuracy-tal m.m.).

**Vigtigt til både elever og undervisere:** Der er med vilje omkring dobbelt så mange
opgaver, som tiden tillader — ingen forventes at nå alt! Opgaver markeret med ⭐ er
til dem, der er foran, og opgaver markeret med 🐛 indeholder *bevidste* fejl, som skal
findes og rettes (så bliv ikke forskrækket, hvis "Run all" fejler dér).

## Sådan åbnes materialet (Google Colab)

Åbn en notebook direkte fra GitHub:

```
https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Intro-ML/<notebook-navn>.ipynb
```

fx [1-Data-manipulation.ipynb](https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Intro-ML/1-Data-manipulation.ipynb).

Alle notebooks kører fint på CPU — der skal **ikke** vælges GPU-runtime.

## Data & Plan B (hvis Kaggle driller)

Notebooks'ene henter selv data fra Kaggle med `kagglehub`:

- Pokémon-stats: [`abcsds/pokemon`](https://www.kaggle.com/datasets/abcsds/pokemon)
- MNIST som CSV: [`oddrationale/mnist-in-csv`](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

Hvis camp-wifi eller Kaggle fejler, ligger der fallback-kopier i `data/`, som kan
indlæses direkte fra GitHub — hver notebook har en synlig "🚨 Plan B"-celle med den
færdige kode. Som plan C til MNIST kan `torchvision.datasets.MNIST` bruges (den er
forudinstalleret i Colab og henter fra et andet mirror end Kaggle).

`hjaelpefunktioner.py` (tre små plottefunktioner — al rigtig kode står i
notebooks'ene) hentes automatisk med en `!wget`-celle. Fejler den, så upload filen
manuelt via Colabs filpanel (📁-ikonet i venstre side).

## Vedligehold

MNIST-fallback-filerne i `data/` er stratificerede subsamples (16000 trænings- og
2000 testrækker, lige mange pr. ciffer, seed 42) af `oddrationale/mnist-in-csv`, genereret med:

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

Notebooks committes med ryddede outputs (ellers vokser repoet vildt — spørg 2025-holdet).
