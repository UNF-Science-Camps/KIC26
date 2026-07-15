# Specialiserede Modeller

Dette er materialet til emnet **Specialiserede Modeller** på KIC26. Temaet er *det
rigtige værktøj til det rigtige job*: I Intro-ML lærte I det generalværktøj, der hedder
neurale netværk — nu møder I seks specialiserede modeller, der hver især er BEDST til
noget bestemt.

**Forudsætning:** Intro-ML-emnet (pandas, PyTorch, `nn.Module`, træningsrytmen).

## Notebooks — vælg en menu!

Notebooks'ene er **uafhængige af hinanden** (hver har sit eget setup og sit eget
datasæt), så I kan frit vælge, hvilke og hvor mange I når. Der er materiale til ca. 9
timer, men typiske hold har 3–6 timer:

| Notebook | Speciale | Datasæt | ca. tid |
|---|---|---|---|
| `1-Beslutningstraeer.ipynb` | tabeldata + forklarlighed | svampe: spiselig eller giftig? 🍄 | 1,5 t |
| `2-NN-udbygninger.ipynb` | overfitting-værktøjskassen | hjertesygdom 🫀 | 1,5 t |
| `3-CNN.ipynb` | billeder | FashionMNIST 👕 (+ CIFAR-10 ⭐) | 2 t |
| `4-Clustering.ipynb` | mønstre UDEN labels | kundesegmentering 🛍️ | 1 t |
| `5-RNN-sekvenser.ipynb` | sekvenser & tid | vejrdata 🌡️ | 1,5 t |
| `6-Autoencoder.ipynb` | kompression & anomalier | håndskrevne tal 🔢 | 1,5 t |

**Menu-forslag:**
- **3 timer:** 1 (træer) + 3 (CNN) — det klassiske makkerpar: tabeller & billeder.
- **4,5 timer:** + 2 (NN-udbygninger) *eller* 4 (clustering).
- **6 timer:** + 4 (clustering) og 5 (RNN) *eller* 6 (autoencoder) — vælg efter holdets smag.

Til hver notebook findes en `*_løsningsforslag.ipynb` med udfyldte svar og lærernoter
(målte accuracy-tal og køretider).

**Vigtigt (elever & undervisere):** Der er med vilje ca. dobbelt så mange opgaver, som
tiden tillader — ingen når alt! ⭐-opgaver er til dem, der er foran; 🐛-opgaver
indeholder *bevidste* fejl, der skal findes (så "Run all" må gerne fejle dér).

## Efter dette emne: campens finale 🚀

Campen slutter med NLP og sprogmodeller: `../LLM/transformer_workshop.ipynb` — byg
videre på en færdigtrænet dansk fantasy-sprogmodel og finetune den. (Bemærk: dén kræver
en T4-GPU i Colab, i modsætning til alt materialet her, som kører på CPU.)

## Sådan åbnes materialet (Google Colab)

```
https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Specialiserede-Modeller/<notebook-navn>.ipynb
```

Alle notebooks kører på CPU — vælg ikke GPU-runtime.

## Data & Plan B (hvis Kaggle driller)

Notebooks'ene henter selv data fra Kaggle med `kagglehub`:

- Svampe: [`uciml/mushroom-classification`](https://www.kaggle.com/datasets/uciml/mushroom-classification)
- Hjertesygdom: [`fedesoriano/heart-failure-prediction`](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- FashionMNIST: [`zalando-research/fashionmnist`](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- Kundedata: [`vjchoudhary7/customer-segmentation-tutorial-in-python`](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- Vejrdata: [`budincsevity/szeged-weather`](https://www.kaggle.com/datasets/budincsevity/szeged-weather)
- MNIST (autoencoder): [`oddrationale/mnist-in-csv`](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)

Fejler Kaggle, ligger der fallback-kopier i `data/` (og MNIST i `../Intro-ML/data/`) —
hver notebook har en synlig "🚨 Plan B"-celle med færdig kode. `hjaelpefunktioner.py`
(fire små plottefunktioner) hentes med en `!wget`-celle; fejler den, uploades filen
manuelt via Colabs filpanel (📁).

## Vedligehold

FashionMNIST-fallbacks er stratificerede subsamples (10000 trænings-/2000 testrækker,
lige mange pr. klasse, seed 42). Vejr-fallbacken er daglige middeltemperaturer
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
dagligt = (vejr.set_index("Formatted Date").sort_index()["Temperature (C)"]
               .resample("D").mean().dropna().round(2).reset_index())
dagligt.columns = ["dato", "temperatur"]
dagligt.to_csv("data/vejr_dagligt.csv.gz", index=False, compression="gzip")
```

Notebooks committes med ryddede outputs.
