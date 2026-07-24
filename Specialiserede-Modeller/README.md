# Specialiserede Modeller

Dette er materialet til emnet **Specialiserede Modeller** på KIC26. Temaet er *det
rigtige værktøj til det rigtige job*: I Intro-ML lærte I det generalværktøj, der hedder
neurale netværk — nu møder I seks specialiserede modeller, der hver især er BEDST til
noget bestemt.

**Forudsætning:** Intro-ML-emnet (pandas, PyTorch, `nn.Module`, træningsrytmen).
Emnekæden på campen: [Introduktion til programmering](../Intro-Programmering/) →
regression → [Intro-ML](../Intro-ML/) → dette emne → [Markov-kæder](../Markov-chain/) →
[LLM-workshoppen](../LLM/).

## Notebooks — vælg en menu!

Notebooks'ene er **uafhængige af hinanden** (hver har sit eget setup og sit eget
datasæt), så I kan frit vælge, hvilke og hvor mange I når, og i hvilken rækkefølge. Der
er materiale til ca. 9 timer, men skemaet giver typisk 3 timer — så det er en menu, ikke
et fast forløb:

| Notebook | Afsnit | Speciale | Datasæt | ca. tid |
|---|---|---|---|---|
| `1-Beslutningstraeer.ipynb` | 1–2 | tabeldata + forklarlighed | svampe: spiselig eller giftig? | 1,5 t |
| `2-NN-udbygninger.ipynb` | 3–4 | overfitting-værktøjskassen | hjertesygdom | 1,5 t |
| `3-CNN.ipynb` | 5–6 | billeder | FashionMNIST (+ CIFAR-10, ekstra) | 2 t |
| `4-Clustering.ipynb` | 7 | mønstre UDEN labels | kundesegmentering | 1 t |
| `5-RNN-sekvenser.ipynb` | 8–9 | sekvenser & tid | vejrdata | 1,5 t |
| `6-Autoencoder.ipynb` | 10–11 | kompression & anomalier | håndskrevne tal | 1,5 t |

**Menu-forslag (3 timer):**
- **Det klassiske makkerpar:** 1 (træer) + 3 (CNN) — tabeller & billeder.
- **Bredden:** 1 (træer) + 4 (clustering) + 5 (RNN) — tre vidt forskellige datatyper på lidt tid.
- **Efter smag:** vælg selv tre — hver notebook er selvstændig.

Til hver notebook findes en `*_løsningsforslag.ipynb` med udfyldte svar og lærernoter
(målte accuracy-tal og køretider).

**Vigtigt (elever & undervisere):** Der er med vilje ca. dobbelt så mange opgaver, som
tiden tillader — ingen når alt! Opgaver mærket **(ekstra)** er til dem, der er foran;
opgaver mærket **(find fejlen)** indeholder *bevidste* fejl, der skal findes (så
"Run all" må gerne fejle dér).

## Efter dette emne: campens finale

Campen slutter med [Markov-kæder](../Markov-chain/) og sprogmodeller:
`../LLM/transformer_workshop.ipynb` — byg videre på en færdigtrænet dansk fantasy-sprogmodel
og finetune den. (Bemærk: dén kræver en T4-GPU i Colab, i modsætning til alt materialet
her, som kører på CPU.)

## Sådan åbnes materialet (Google Colab)

```
https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Specialiserede-Modeller/<notebook-navn>.ipynb
```

Alle notebooks kører på CPU — vælg ikke GPU-runtime.

## Data & Plan B

Alle data og `hjaelpefunktioner.py` hentes med en `!wget`-celle fra GitHub — intet
Kaggle-login og (næsten) intet `pip install` kræves. Filerne ligger i `data/` (MNIST til
autoencoderen genbruges fra `../Intro-ML/data/`).

**Plan B:** Fejler `wget`-cellen, så upload den nævnte fil manuelt via Colabs filpanel
(mappeikonet til venstre) — filerne ligger i `Specialiserede-Modeller/data` på GitHub.
`hjaelpefunktioner.py` (fire små plottefunktioner) hentes på samme måde.

**Undtagelse:** CIFAR-10 (den ekstra farvebillede-opgave i `3-CNN.ipynb`) er for stor
til repoet (~170 MB) og hentes derfor stadig med `kagglehub`/`torchvision`.

## Vedligehold

FashionMNIST-filerne er stratificerede subsamples (10000 trænings-/2000 testrækker,
lige mange pr. klasse, seed 42). Vejr-filen er daglige middelværdier (temperatur + fugt)
resamplet fra timedata. Genskabes med:

```python
import kagglehub, pandas as pd

sti = kagglehub.dataset_download("zalando-research/fashionmnist")
for fil, n, ud in [("fashion-mnist_train.csv", 10000, "fashion_traen_lille.csv.gz"),
                   ("fashion-mnist_test.csv", 2000, "fashion_test_lille.csv.gz")]:
    hel = pd.read_csv(f"{sti}/{fil}")
    dele = [hel[hel["label"] == c].sample(n=n // 10, random_state=42) for c in range(10)]
    pd.concat(dele).sample(frac=1, random_state=42).reset_index(drop=True).to_csv(
        f"data/{ud}", index=False, compression="gzip")

sti = kagglehub.dataset_download("budincsevity/szeged-weather")
vejr = pd.read_csv(f"{sti}/weatherHistory.csv")
vejr["Formatted Date"] = pd.to_datetime(vejr["Formatted Date"], utc=True)
g = vejr.set_index("Formatted Date").sort_index()
dagligt = pd.DataFrame({"temperatur": g["Temperature (C)"].resample("D").mean(),
                        "fugt": g["Humidity"].resample("D").mean()}).dropna().reset_index(drop=True)
dagligt.to_csv("data/vejr_dagligt.csv.gz", index=False, compression="gzip")
```

Notebooks committes med ryddede outputs.
