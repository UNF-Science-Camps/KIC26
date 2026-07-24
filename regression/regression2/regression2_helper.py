"""
Landskaber + evalueringsharness til træningsloop-blokkens gradient-descent-konkurrence.
Eleverne skriver deres egen gd_metode(loss, gradient, start, ...) -> (a,b) — typisk versioneret
som egen_gd_metode_v1, v2, ... i takt med at de bygger flere idéer ind (momentum, RMSprop, ...)
— og sammenligner en HEL LISTE af metoder ad gangen med evaluer_gd_metode([("gd", gd_metode), ...]),
kørt på tværs af 4 landskaber × 4 udvalgte startpunkter.
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utilities'))
import numpy as np
import sesy_vizualisation as sesy_viz


class BudgetOverskredet(Exception):
    """Rejses når en gd_metode-funktion har brugt alle sine kald til loss/gradient."""
    pass


# ── Landskabernes matematik ──────────────────────────────────────────────
# hvert landskab er loss(a,b) + gradient(a,b) -> (da,db), akkurat som regressions egen
# opg3_2/opg8_1 — bare 4 forskellige "flader" i stedet for kun lineær regressions.

def _generer_linje_punkter(n=100, a_sand=1.2, b_sand=1.5, x_range=(0, 10), stoej=0.6, seed=42):
    """~100 punkter normalfordelt omkring en linje — landskabet til minibatch/SGD-afsnittet,
    og konkurrencens 'linjefitting'-landskab."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*x_range, n)
    ys = a_sand * xs + b_sand + rng.normal(0, stoej, n)
    return list(zip(xs.tolist(), ys.tolist()))


LINJE_PUNKTER = _generer_linje_punkter()


def _linje_loss(a, b, punkter=LINJE_PUNKTER):
    n = len(punkter)
    return sum((a * x + b - y) ** 2 for x, y in punkter) / n


def _linje_gradient(a, b, punkter=LINJE_PUNKTER):
    n = len(punkter)
    da, db = 0.0, 0.0
    for x, y in punkter:
        fejl = a * x + b - y
        da += 2 * fejl * x
        db += 2 * fejl
    return da / n, db / n


def _linje_minimum(punkter=LINJE_PUNKTER):
    """Normalligningerne — samme lukkede løsning som regression_facit."""
    xs = np.array([p[0] for p in punkter])
    ys = np.array([p[1] for p in punkter])
    A = np.column_stack([xs, np.ones_like(xs)])
    a, b = np.linalg.inv(A.T @ A) @ A.T @ ys
    return float(a), float(b)


# offentlige navne (uden understreg) til minibatch-afsnittet — "givet" infrastruktur, ligesom
# sesy_viz's egne funktioner: I skal ikke selv udlede MSE-gradienten igen, den kender I allerede
# fra regression — brug den bare på et TILFÆLDIGT UDPLUK af punkter i stedet for alle.
linje_loss = _linje_loss
linje_gradient = _linje_gradient


def _plateau_led(t, k):
    """Ét led i plateau-familien (jf. Refference/SOP/Gradient Decent's PlatauExample)
    — typisk 2 lokale minima langs denne akse."""
    return t ** 3 + k * t + 0.5 * t ** 4


def _plateau_led_grad(t, k):
    return 3 * t ** 2 + k + 2 * t ** 3


def _platau1d_loss(a, b):
    # "1D": den eneste interessante (svære) akse er a — b-aksen er en simpel skål (b²) der
    # trækker mod midterlinjen b=0, så landskabet ikke bare er en uendelig flad rygrad langs b.
    return _plateau_led(a, -0.25) + b ** 2


def _platau1d_gradient(a, b):
    return _plateau_led_grad(a, -0.25), 2 * b


def _platau2d_loss(a, b):
    return _plateau_led(a, -0.25) + _plateau_led(b, -0.8)


def _platau2d_gradient(a, b):
    return _plateau_led_grad(a, -0.25), _plateau_led_grad(b, -0.8)


def _rosenbrock_loss(a, b):
    return (1 - a) ** 2 + 100 * (b - a ** 2) ** 2


def _rosenbrock_gradient(a, b):
    da = -2 + 2 * a - 400 * (-a ** 2 + b) * a
    db = -200 * a ** 2 + 200 * b
    return da, db


# ── Registrering: her tilføjes nye landskaber ────────────────────────────

def _multistart_minimum(loss, gradient, view, forsoeg=40, skridt=3000, lr=0.01, seed=0):
    """
    Finder (tilnærmet) det globale minimum ved almindelig gradient descent fra mange
    tilfældige startpunkter, og beholder det bedste resultat. Køres ÉN gang ved import,
    ikke en del af elevernes kald-budget — bruges kun når 'minimum' ikke er kendt analytisk.
    """
    rng = np.random.default_rng(seed)
    a_min, a_max, b_min, b_max = view
    bedste = (None, None, np.inf)
    for _ in range(forsoeg):
        a = rng.uniform(a_min, a_max)
        b = rng.uniform(b_min, b_max)
        for _ in range(skridt):
            da, db = gradient(a, b)
            a -= lr * da
            b -= lr * db
            if not (np.isfinite(a) and np.isfinite(b)):
                break
        l = loss(a, b)
        if np.isfinite(l) and l < bedste[2]:
            bedste = (a, b, l)
    return bedste[0], bedste[1]


def _byg_landskab(navn, loss, gradient, a_range, b_range, minimum=None):
    """minimum: giv (a*,b*) hvis den kendes analytisk — ellers findes den selv (se
    _multistart_minimum). Sådan tilføjes et nyt landskab: bare loss+gradient+view."""
    if minimum is None:
        minimum = _multistart_minimum(loss, gradient, (*a_range, *b_range))
    a_star, b_star = minimum
    return {
        "navn": navn, "loss": loss, "gradient": gradient,
        "a_range": a_range, "b_range": b_range,
        "minimum": (a_star, b_star), "min_loss": float(loss(a_star, b_star)),
    }


_platau1d_a_star, _ = _multistart_minimum(_platau1d_loss, _platau1d_gradient, (-2.1, 1.1, -2.1, 1.1))

LANDSKABER = [
    _byg_landskab("linjefitting", _linje_loss, _linje_gradient, (-1, 3), (-3, 3), minimum=_linje_minimum()),
    # b-view centreret om 0 (b²-skålens midterlinje) — anderledes range end a, som er valgt til
    # at ramme a-aksens 2 lokale minima. b-aksens minimum er PRÆCIST 0.0, låst eksplicit i
    # stedet for at stole på multistart-søgningens numeriske tilnærmelse for den akse.
    _byg_landskab("plateau (1D)", _platau1d_loss, _platau1d_gradient, (-2.1, 1.1), (-1, 1), minimum=(_platau1d_a_star, 0.0)),
    _byg_landskab("plateau (2D)", _platau2d_loss, _platau2d_gradient, (-2.1, 1.1), (-2.1, 1.1)),
    _byg_landskab("rosenbrock", _rosenbrock_loss, _rosenbrock_gradient, (-1.5, 1.5), (-1, 2), minimum=(1.0, 1.0)),
]


# ── 4 udvalgte startpunkter pr. landskab ──────────────────────────────────
# HÅNDPLUKKEDE (ikke et generisk hjørne-mønster) — valgt ved faktisk at køre gd/momentum/
# rmsprop/adam fra mange kandidat-punkter og beholde de 4 pr. landskab der viser mest
# forskellig opførsel: fx et sted alle metoder er enige, et sted de er dybt uenige (en metode
# vinder klart), osv. Se intro-ml-plan-memoen for den fulde udforskning bag disse tal.

INTERESSANTE_STARTPUNKTER = {
    "linjefitting":  [(-0.6, -2.4), (1.0, 0.0), (2.2, 1.8), (1.6, 0.9)],
    "plateau (1D)":  [(-1.78, -0.8), (1.0, 0.6), (-0.5, 0.0), (-0.3, -0.8)],
    "plateau (2D)":  [(-1.78, 0.78), (0.78, -1.78), (-0.5, -0.5), (-0.5, -1.78)],
    "rosenbrock":    [(-1.0, -0.5), (0.0, 0.0), (1.2, 1.7), (0.0, -0.9)],
}


# ── Instrumentering: tæller og logger kald til loss/gradient ────────────

def _sikker_loss(fn, a, b):
    """Regner fn(a,b) med numpy-tal, så en eksploderet metode (fx Rosenbrock med for stort lr)
    giver et meget dårligt tal (inf/nan) i stedet for at vælte hele konkurrencen med en exception."""
    with np.errstate(all="ignore"):
        try:
            return float(fn(np.float64(a), np.float64(b)))
        except (OverflowError, ValueError):
            return float("nan")


def _sikker_gradient(fn, a, b):
    with np.errstate(all="ignore"):
        try:
            da, db = fn(np.float64(a), np.float64(b))
            return float(da), float(db)
        except (OverflowError, ValueError):
            return float("nan"), float("nan")


_AFSTAND_GRAENSE = 1000  # et kald om et punkt længere væk end dette fra landskabets midte ansés som useriøst


def instrumenter(landskab, budget=100, afstand_graense=_AFSTAND_GRAENSE):
    """
    Pakker landskabets loss/gradient ind. loss og gradient tælles og huskes (positioner
    undersøgt) HVER FOR SIG, men deler ét fælles budget på i alt `budget` kald.

    Stopper kørslen (rejser BudgetOverskredet, med en forklarende print) i 2 tilfælde:
    - I har brugt alle jeres `budget` kald tilsammen — jeres svar bliver automatisk det
      sidste punkt I nåede at undersøge (så I godt må stoppe FØR budgettet er brugt, ved
      selv at returnere — det er kun hvis I IKKE selv stopper i tide, at dette griber ind).
    - I beder om et punkt mere end `afstand_graense` fra landskabets midte — det regnes
      slet ikke på (undgår at meningsløst store tal vælter jeres egen udregning), og jeres
      svar bliver det sidste GYLDIGE punkt I undersøgte før det.
    """
    a0, a1 = landskab["a_range"]
    b0, b1 = landskab["b_range"]
    centrum_a, centrum_b = (a0 + a1) / 2, (b0 + b1) / 2

    loss_punkter, gradient_punkter = [], []
    tilstand = {"antal": 0, "sidste": None, "for_langt_vaek": False}

    def _brug_et_kald(a, b):
        afstand = ((a - centrum_a) ** 2 + (b - centrum_b) ** 2) ** 0.5
        if afstand > afstand_graense:
            print(f"⚠ punktet ({a:.1f}, {b:.1f}) er {afstand:.0f} fra landskabets midte — "
                  f"det virker useriøst langt væk, stopper her.")
            tilstand["for_langt_vaek"] = True
            raise BudgetOverskredet("punkt for langt fra landskabet")
        tilstand["antal"] += 1
        if tilstand["antal"] > budget:
            print(f"⚠ I har brugt alle jeres {budget} kald til loss/gradient tilsammen — "
                  f"bruger punkt nr. {budget} som jeres svar.")
            raise BudgetOverskredet("budget opbrugt")

    def loss(a, b):
        _brug_et_kald(a, b)
        v = _sikker_loss(landskab["loss"], a, b)
        loss_punkter.append((a, b, v))
        tilstand["sidste"] = (a, b)
        return v

    def gradient(a, b):
        _brug_et_kald(a, b)
        v = _sikker_gradient(landskab["gradient"], a, b)
        gradient_punkter.append((a, b, v))
        tilstand["sidste"] = (a, b)
        return v

    return loss, gradient, loss_punkter, gradient_punkter, tilstand


def koer_en_gang(gd_metode, landskab, start, budget=100):
    """
    Kør gd_metode(loss, gradient, start) ÉN gang på ét landskab, fra ét startpunkt. Returnerer
    et resultat-dict (landskab, start, slut, slut_loss, bedste_loss, loss_punkter,
    gradient_punkter) — selve visualiseringen af det bygger I i notebook'en.
    """
    loss, gradient, loss_punkter, gradient_punkter, tilstand = instrumenter(landskab, budget)
    try:
        slut = gd_metode(loss, gradient, start)
    except BudgetOverskredet:
        slut = tilstand["sidste"] or start
    a, b = slut
    slut_loss = _sikker_loss(landskab["loss"], a, b)
    return {
        "landskab": landskab["navn"],
        "a_range": landskab["a_range"], "b_range": landskab["b_range"],
        "start": start, "slut": (float(a), float(b)),
        "slut_loss": slut_loss,
        "bedste_loss": landskab["min_loss"],
        # True hvis metoden endte langt uden for landskabet (uanset om slut_loss selv er et
        # stort-men-endeligt tal eller nan/inf) — se instrumenter()'s afstands-tjek.
        "eksploderet": bool(tilstand["for_langt_vaek"]) or not np.isfinite(slut_loss),
        "loss_punkter": loss_punkter, "gradient_punkter": gradient_punkter,
    }


# ── Farver: fast pr. kendt metode, så gd/momentum/rmsprop/adam altid ser ens ud ──────────

_PALET = {
    "blaa": "#2a78d6", "groen": "#008300", "magenta": "#e87ba4", "gul": "#eda100",
    "cyan": "#1baf7a", "orange": "#eb6834", "lilla": "#4a3aa7", "roed": "#e34948",
}
_KENDTE_METODE_FARVER = {
    "gd": _PALET["blaa"], "momentum": _PALET["groen"],
    "rmsprop": _PALET["magenta"], "adam": _PALET["gul"],
}
_EKSTRA_FARVE_REKKEFOLGE = [_PALET["cyan"], _PALET["orange"], _PALET["lilla"], _PALET["roed"]]


def _tildel_farver(navne):
    """Fast farve for gd/momentum/rmsprop/adam (samme farve overalt i notebooken) — egne,
    ukendte metode-navne får en farve fra en fast rækkefølge, i den rækkefølge de optræder."""
    farver, næste = {}, 0
    for navn in navne:
        if navn in farver:
            continue
        if navn in _KENDTE_METODE_FARVER:
            farver[navn] = _KENDTE_METODE_FARVER[navn]
        else:
            farver[navn] = _EKSTRA_FARVE_REKKEFOLGE[næste % len(_EKSTRA_FARVE_REKKEFOLGE)]
            næste += 1
    return farver


# ── Sammenlign en liste af metoder på hele konkurrencen, i ét kald ───────

def evaluer_gd_metode(metoder, landskaber=LANDSKABER, budget=100, resolution=30):
    """
    Kør en eller flere metoder på tværs af alle landskaber og deres 4 udvalgte startpunkter,
    og vis resultatet som ét grid — én række pr. landskab. Første søjle i hver række er et
    søjlediagram (median slut-loss pr. metode + bedste mulige loss som stiplet linje), de næste
    4 paneler viser alle metoders sti overlejret på samme landskab, ét panel pr. startpunkt.

    metoder: liste af (navn, funktion)-par, fx [("gd", gd_metode), ("momentum", momentum_metode)]
    — skriv hele listen igen (med alt I vil sammenligne) hver gang I kalder denne, i stedet for
    at bygge videre på en gammel liste.
    """
    farver = _tildel_farver([navn for navn, _ in metoder])
    paneler, row_labels = [], []

    for landskab in landskaber:
        row_labels.append(landskab["navn"])
        starts = INTERESSANTE_STARTPUNKTER[landskab["navn"]]
        resultater = {navn: [koer_en_gang(fn, landskab, start, budget) for start in starts]
                      for navn, fn in metoder}

        # afstand til bedste mulige loss, én liste pr. metode — én værdi pr. udvalgt startpunkt
        # (samme rækkefølge som sti-panelerne herunder), None hvis den kørsel eksploderede.
        afstande = [
            [None if r["eksploderet"] else r["slut_loss"] - landskab["min_loss"] for r in resultater[navn]]
            for navn, _ in metoder
        ]

        # bar-panelet er den eneste farve/navn-forklaring i rækken (labeled x-akse) — de 4
        # sti-paneler til højre for det holder sig derfor bevidst UDEN egen legend, for ikke at
        # gentage samme 4 navne i alle 20 paneler i gridet.
        paneler.append(sesy_viz.bar_sammenligning(
            [navn for navn, _ in metoder], afstande, [farver[navn] for navn, _ in metoder],
        ))

        for i, start in enumerate(starts):
            stier = []
            for navn, _ in metoder:
                r = resultater[navn][i]
                sti = ([(a, b) for (a, b, _) in r["gradient_punkter"]]
                       or [(a, b) for (a, b, _) in r["loss_punkter"]] or [start])
                stier.append((sti, farver[navn]))
            paneler.append(sesy_viz.loss_kontur(
                landskab["loss"], landskab["a_range"], landskab["b_range"], resolution=resolution,
                ekstra_stier=stier, colorbar=False, titel=f"start=({start[0]:.2g}, {start[1]:.2g})",
            ))

    sesy_viz.display_grid(paneler, cols=5, row_labels=row_labels)
