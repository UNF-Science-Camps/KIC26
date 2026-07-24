# Pachinko simulator

A pachinko / Galton-board simulation written in Python. A coin dropped on the
apex peg bounces **left or right with a 50/50 chance** at every peg; after 10
bounces it lands in one of 11 bins, building up a binomial bell curve.

Features:

- **Drop coins** one at a time (animated in the browser) or 100 at once.
- **Step mode** — advance the coin one bounce at a time and inspect each decision.
- **Show %** — green arrows at the coin's current vertex show the two possible
  next vertices, each labelled **50%**; the theoretical landing share is printed
  under every bin.
- **Last vertex / last edge** — a red edge and red ring mark the move the coin
  made last; the status line spells it out as `(row, pos)`. By default only
  that last edge is drawn; the **Full path** toggle shows the whole travelled
  path instead.
- **Heatmap** — a toggle that overlays the paths of *all* coins dropped so
  far: every edge is coloured light→dark blue by how many coins travelled it
  (few ↔ many legend in the corner), so the 50/50 traffic pattern builds up
  visibly. Reset clears it along with the bins.

## Run it in the browser directly

Open **`pachinko_browser.html`** — double-click it, or serve the folder:

```sh
python3 -m http.server   # then visit http://localhost:8000/pachinko_browser.html
```

All the app logic is Python, executed in the browser by
[PyScript](https://pyscript.net) — the first visit downloads the runtime from
the PyScript CDN (internet required once; it is cached afterwards).

Controls: **Drop coin**, **Next bounce** (step mode), **Finish drop**,
**Drop 100**, **Reset**, the **Step/Auto** mode switch and the **Show %**
toggle. In step mode, **Space**/**N** or a click on the board also advances one
bounce, and **D** drops a coin. In Auto mode the coin runs on its own, lingering
briefly at each peg to show the 50/50 arrows while Show % is on. The page
follows your light/dark system theme.

## Run it in a notebook

```sh
jupyter lab    # or: jupyter notebook
```

Open **`pachinko_notebook.ipynb`** and run the cells. Requirements:
`matplotlib` (needed) and `ipywidgets` (optional — for the button UI; a
manual re-run-this-cell step-through is included that needs no widgets).

## Files

| File | Purpose |
|---|---|
| `pachinko.py` | Core simulation (pure stdlib). `python3 pachinko.py` prints a 2000-drop ASCII histogram. |
| `pachinko_mpl.py` | Matplotlib renderer used by the notebook. |
| `pachinko_notebook.ipynb` | Notebook UI (widgets + manual stepping). |
| `pachinko_browser.html` | Self-contained PyScript app; embeds a copy of the `pachinko.py` model — keep the two in sync if you edit the physics. |

The maths: a coin ending in bin *k* made exactly *k* right-bounces out of 10,
so `P(bin k) = C(10, k) / 2**10` — that is what `PachinkoBoard.bin_probability`
returns and what the percentages under the bins show.
