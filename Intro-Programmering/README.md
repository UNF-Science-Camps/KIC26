# Introduktion til programmering

Dette er materialet til emnet **Introduktion til programmering** på KIC26 — campens
allerførste emne. Der forudsættes **ingen** programmeringserfaring: vi starter ved
`print("Hej")` og slutter med klasser, f-strings og en hjemmebygget tokenizer. Emnet
er ca. 9 timers kerneundervisning (tre notebooks à 3 timer) plus en kort bro-notebook,
og det bygger op til regressionsemnet og [Intro til Machine Learning](../Intro-ML/).

## Notebooks (kør dem i rækkefølge)

| # | Notebook | Indhold | ca. tid |
|---|---|---|---|
| 1 | `1-Foerste-skridt.ipynb` | print, variabler, matematik og if/else — de allerførste skridt | 3 t |
| 2 | `2-Loekker-lister-og-funktioner.ipynb` | lister, for-løkker, funktioner og numpy | 3 t |
| 3 | `3-Plots-klasser-og-tekst.ipynb` | matplotlib, klasser, f-strings og ordbøger (med tokenizer-finale!) | 3 t |
| 4 | `4-Kaggle-og-kurvefit.ipynb` | kagglehub, `np.loadtxt` og `curve_fit` — broen til regressionsemnet | 1,5 t |
| 5 | `5-Ekstra-opgaver.ipynb` | store kombinerede opgaver (E.1–E.9) til de hurtige | overskud |

Til hver notebook findes en `*_løsningsforslag.ipynb` med udfyldte svar og lærernoter
(målte fit-parametre m.m.).

**Vigtigt til både elever og undervisere:** Der er med vilje omkring dobbelt så mange
opgaver, som tiden tillader — ingen forventes at nå alt! Opgaver markeret med ⭐ er
til dem, der er foran, og opgaver markeret med 🐛 indeholder *bevidste* fejl, som skal
findes og rettes (så bliv ikke forskrækket, hvis "Run all" fejler dér). Opgaverne er
desuden trappet: eleverne starter med at *ændre* i kode, der virker, udfylder derefter
`...`-huller — og skriver gradvist mere og mere selv; ingen opgave starter fra en
helt blank celle.

## Sådan åbnes materialet (Google Colab)

Åbn en notebook direkte fra GitHub:

```
https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Intro-Programmering/<notebook-navn>.ipynb
```

fx [1-Foerste-skridt.ipynb](https://colab.research.google.com/github/UNF-Science-Camps/KIC26/blob/main/Intro-Programmering/1-Foerste-skridt.ipynb).

Alle notebooks kører fint på CPU — der skal **ikke** vælges GPU-runtime.

## Data & Plan B (hvis Kaggle driller)

Notebook 1–3 bruger ingen datafiler. Notebook 4 henter selv data fra Kaggle med
`kagglehub`:

- Fiskemarkedet: [`aungpyaeap/fish-market`](https://www.kaggle.com/datasets/aungpyaeap/fish-market)

Hvis camp-wifi eller Kaggle fejler, ligger der en fallback-kopi i `data/Fish.csv`,
som kan indlæses direkte fra GitHub — notebooken har en synlig "🚨 Plan B"-celle med
den færdige kode.

Øvefilerne `dat1.txt`, `dat2.txt` og `dat3.txt` (til plots og kurve-fit i notebook 4
og 5) hentes automatisk med en `!wget`-celle fra `data/`-mappen her. Fejler den, så
upload filerne manuelt via Colabs filpanel (📁-ikonet i venstre side).

## Vedligehold

`dat1.txt`–`dat3.txt` er uændrede kopier af de gamle øvefiler fra
`MLC2025/Introduktion til programmering/` (1000 rækker `x y` pr. fil; dat1 er lineær,
dat2 en parabel og dat3 eksponentiel — de målte fit-parametre står i
løsningsforslaget til notebook 4).

`data/Fish.csv` er en uændret kopi af Kaggle-datasættet, som kan regenereres med:

```python
import kagglehub, shutil
sti = kagglehub.dataset_download("aungpyaeap/fish-market")
shutil.copy(f"{sti}/Fish.csv", "data/Fish.csv")
```

Notebooks committes med ryddede outputs (ellers vokser repoet vildt — spørg 2025-holdet).
