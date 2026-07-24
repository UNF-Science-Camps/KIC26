import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '98-Helpers'))
try:
    import helpers as _viz
except ImportError:
    _viz = None


def _ok(name):
    print(f"Alle tests bestået for {name}  ✓")


def _fail(name, besked):
    print(f"{name}: {besked}  ✗")


def _eq(got, exp, tol=1e-9):
    """Float-tolerant lighed for enkeltværdi eller tuple."""
    if isinstance(exp, tuple):
        return len(got) == len(exp) and all(
            abs(float(g) - float(e)) < tol for g, e in zip(got, exp)
        )
    return abs(float(got) - float(exp)) < tol


# ── opg 1 (minibatch: træk et tilfældigt udpluk af punkter) ─────────────────
# NB: resultatet er tilfældigt, så vi tjekker EGENSKABER (rigtig størrelse, ingen
# gentagelser, kun punkter der faktisk findes i datasættet) i stedet for eksakte værdier.

def test_ex1_1(func):
    """træk k punkter, UDEN tilbagelægning, fra en liste af punkter"""
    name = "opg1_1"
    points = [(float(i), float(i) * 2) for i in range(20)]
    for k in [1, 5, 20]:
        for _ in range(25):  # mange forsøg — fanger fejl der kun rammer nogle gange (fx tilbagelægning)
            try:
                batch = func(points, k)
            except Exception as e:
                _fail(name, f"jeres funktion fejlede med k={k}: {e}")
                return
            if len(batch) != k:
                _fail(name, f"forventede {k} punkter tilbage, fik {len(batch)} (k={k})")
                print("  Hint: batchen skal have PRÆCIS k punkter — tjek at I bruger k, ikke et fast tal.")
                return
            if len(set(batch)) != len(batch):
                _fail(name, f"batchen indeholder samme punkt flere gange (k={k}) — træk UDEN tilbagelægning")
                print("  Hint: brug random.sample(punkter, k), ikke random.choice i et loop.")
                return
            if not set(batch).issubset(set(points)):
                _fail(name, f"batchen indeholder punkter der slet ikke er i den oprindelige liste (k={k})")
                print("  Hint: batchen skal bestå af punkter FRA punkter — ikke nye eller ændrede punkter.")
                return
    _ok(name)


# ── opg 1 (fortsat): sæt opg1_1 sammen med den givne skridt-funktion fra recap ──

def test_ex1_2(func):
    """træningsloop: træk en NY batch (opg1_1) og tag ét skridt (skridt, fra recap), n gange i træk.
    Testet med batch_størrelse = ALLE punkterne — så batchen er den samme uanset hvilken
    rækkefølge den trækkes i, og resultatet bliver deterministisk (almindelig, ikke-stokastisk
    gradient descent)."""
    name = "opg1_2"
    points = [(1, 2), (3, 4), (4, 3), (2.5, 2.5)]
    cases = [
        ((0.0, 0.0, points, 4, 0.05, 1),                        (0.80625, 0.28750000000000003)),
        ((0.0, 0.0, points, 4, 0.05, 3),                        (0.890269775390625, 0.3558129882812501)),
        ((0.0, 0.0, points, 4, 0.05, 10),                       (0.8515188432754992, 0.47661894571251673)),
        ((1.0, 1.0, [(0, 0), (1, 2), (2, 4)], 3, 0.1, 5),        (1.3804628806584363, 0.844472098765432)),
    ]
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: kald jeres opg1_1 for at trække en ny batch, og den givne skridt-funktion fra recap — n gange i træk.")
            return
    _ok(name)


def test_ex1_3(tekst, min_ord=15):
    """Ikke en facit-tjekker — der findes intet 'rigtigt' svar her. Tjekker kun at der
    rent faktisk er skrevet noget (nok ord til at være en rigtig beskrivelse, ikke bare
    et par stikord), så I ikke ved et uheld render forbi opgaven."""
    name = "opg1_3"
    ord = tekst.split()
    if len(ord) < min_ord:
        _fail(name, f"kun {len(ord)} ord skrevet — beskriv med en rigtig sætning eller to (mindst {min_ord} ord)")
        print("  Hint: ikke facit-tjekket — prøv batch_størrelse = 1, 5, 20, 50, 100 i cellen ovenfor, og beskriv med egne ord hvad I ser ændre sig.")
        return
    _ok(name)


# ── GD-metoder (gd_metode / momentum_metode / rmsprop_metode) ───────────────────────────
# Alle 3 skal have signaturen metode(loss, start, ..., n=99) -> (a, b), så de kan plugges
# direkte ind i konkurrencens evaluer_gd_metode (som kalder dem uden ekstra argumenter, og
# derfor er afhængige af at jeres egne parametre — lr/beta/n — har defaults). Metoderne får
# KUN loss — gradienten finder de selv med .backward() (jeres gradienten-hjælper).
# test_gd_metode/test_momentum_metode/test_rmsprop_metode tjekker hvilken som helst funktion
# I giver dem (den behøver ikke hedde noget bestemt) — men til fri leg bagefter (jeres egne
# idéer, ingen facit at teste imod) navngiv dem egen_gd_metode_v1, v2, ... og brug
# tk.evaluer_gd_metode(...) direkte i stedet.
# Testet på 2 små, kendte landskaber (IKKE konkurrencens egne — dem skal I stadig selv opdage),
# med et par forskellige startpunkter/hyperparametre, så flere slags fejl bliver fanget.
# Facit-tal er hentet ved at køre en reference-implementation af hver metodes formel.

def _test_loss1(a, b):
    return (a - 2) ** 2 + (b + 1) ** 2


def _test_loss2(a, b):
    return 5 * (a - 1) ** 2 + 0.2 * (b - 3) ** 2


# Autograd af _test_loss1/2 giver præcis de gradienter metoderne skal finde med .backward():
#   d/d(a,b) _test_loss1 = (2(a-2), 2(b+1)),   d/d(a,b) _test_loss2 = (10(a-1), 0.4(b-3)).
# Facit-tallene nedenfor er derfor uændrede fra dengang metoderne fik gradienten foræret.


def _test_method(name, func, cases, hint=None):
    for loss_fn, start, kwargs, exp in cases:
        try:
            got = func(loss_fn, start, **kwargs)
        except Exception as e:
            _fail(name, f"jeres funktion fejlede med start={start}, {kwargs}: {e}")
            return False
        if not _eq(got, exp):
            _fail(name, f"start={start}, {kwargs}  →  {got}  ≠  {exp}")
            if hint:
                print(f"  Hint: {hint}")
            if _viz:
                _viz.feedback_method(loss_fn, start, got, exp)
            return False
    _ok(name)
    return True


def test_gd_step(func):
    """gd_skridt(a, b, loss, lr) — ét skridt: find gradienten af loss med .backward(), træk lr*gradienten fra."""
    cases = [
        ((0.0, 0.0, _test_loss1, 0.1), (0.4, -0.2)),
        ((5.0, -5.0, _test_loss2, 0.05), (3.0, -4.84)),
    ]
    name = "gd_skridt"
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: find gradienten (da, db) af loss med .backward(), gang begge med lr, og træk det fra: a - lr*da, b - lr*db.")
            a, b, loss_fn, lr = inp
            if _viz: _viz.feedback_method(loss_fn, (a, b), got, exp)
            return
    _ok(name)


def test_gd_method(func):
    """gd_metode(loss, start, lr=0.01, n=99) — almindelig gradient descent."""
    cases = [
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, n=5), (1.34464, -0.67232)),
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, n=20), (1.976941569907863, -0.9884707849539315)),
        (_test_loss2, (5.0, -5.0), dict(lr=0.05, n=10), (1.00390625, -3.5365824551003757)),
    ]
    _test_method("gd_metode", func, cases, hint="kald jeres egen gd_skridt n gange i træk.")


def test_momentum_v_update(func):
    """momentum_v_opdater(va, vb, da, db, beta) — det vægtede glidende gennemsnit af gradienten."""
    cases = [
        ((0.0, 0.0, -4.0, 2.0, 0.9), (-0.3999999999999999, 0.19999999999999996)),
        ((1.0, 2.0, 0.5, -1.0, 0.8), (0.9, 1.4000000000000001)),
        ((-0.4, 0.2, 3.0, -1.0, 0.9), (-0.06000000000000011, 0.08000000000000004)),
    ]
    name = "momentum_v_opdater"
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: v er et glidende gennemsnit — beta*v_gammel + (1-beta)*gradient, regnet for a og b hver for sig.")
            return
    _ok(name)


def test_momentum_step(func):
    """momentum_skridt(a, b, va, vb, loss, lr, beta) -> (a, b, va, vb) — ét momentum-skridt."""
    cases = [
        ((0.0, 0.0, 0.0, 0.0, _test_loss1, 0.1, 0.9),
         (0.039999999999999994, -0.019999999999999997, -0.3999999999999999, 0.19999999999999996)),
        ((0.04, -0.02, -0.4, 0.2, _test_loss1, 0.1, 0.9),
         (0.1152, -0.0576, -0.752, 0.376)),
        ((5.0, -5.0, 0.0, 0.0, _test_loss2, 0.05, 0.9),
         (4.8, -4.984, 3.999999999999999, -0.31999999999999995)),
    ]
    name = "momentum_skridt"
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: find gradienten med .backward(), opdater v med jeres momentum_v_opdater, brug SÅ v (ikke gradienten) til skridtet.")
            a, b, loss_fn = inp[0], inp[1], inp[4]
            if _viz: _viz.feedback_method(loss_fn, (a, b), got[:2], exp[:2])
            return
    _ok(name)


def test_momentum_method(func):
    """momentum_metode(loss, start, lr=0.02, beta=0.9, n=99)."""
    cases = [
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, beta=0.9, n=5), (0.5013670144, -0.2506835072)),
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, beta=0.5, n=10), (1.8669852288000002, -0.9334926144000001)),
        (_test_loss2, (5.0, -5.0), dict(lr=0.05, beta=0.9, n=10), (-0.24369164267851545, -4.349386327817435)),
    ]
    _test_method("momentum_metode", func, cases,
                 hint="kald jeres egen momentum_skridt n gange i træk, og husk at bære v videre mellem kaldene.")


def test_rmsprop_s_update(func):
    """rmsprop_s_opdater(sa, sb, da, db, beta) — det glidende gennemsnit af gradienten I ANDEN."""
    cases = [
        ((0.0, 0.0, -4.0, 2.0, 0.9), (1.5999999999999996, 0.3999999999999999)),
        ((2.0, 0.5, 3.0, -1.0, 0.5), (5.5, 0.75)),
    ]
    name = "rmsprop_s_opdater"
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: s er samme idé som momentums v, men af GRADIENTEN I ANDEN — beta*s_gammel + (1-beta)*gradient².")
            return
    _ok(name)


def test_rmsprop_step(func):
    """rmsprop_skridt(a, b, sa, sb, loss, lr, beta, eps) -> (a, b, sa, sb) — ét rmsprop-skridt."""
    cases = [
        ((0.0, 0.0, 0.0, 0.0, _test_loss1, 0.1, 0.9, 1e-8),
         (0.316227763516838, -0.316227761016838, 1.5999999999999996, 0.3999999999999999)),
        ((0.31622776601683794, -0.31622776601683794, 1.6, 0.4, _test_loss1, 0.1, 0.9, 1e-8),
         (0.5261246851931408, -0.5011293906031297, 2.574035574373059, 0.5470177871865296)),
        ((5.0, -5.0, 0.0, 0.0, _test_loss2, 0.05, 0.9, 1e-8),
         (4.841886117116581, -4.841886118554081, 159.99999999999997, 1.024)),
    ]
    name = "rmsprop_skridt"
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: find gradienten med .backward(), opdater s med jeres rmsprop_s_opdater, skalér SÅ gradienten med sqrt(s)+eps.")
            a, b, loss_fn = inp[0], inp[1], inp[4]
            if _viz: _viz.feedback_method(loss_fn, (a, b), got[:2], exp[:2])
            return
    _ok(name)


def test_rmsprop_method(func):
    """rmsprop_metode(loss, start, lr=0.01, beta=0.9, eps=1e-8, n=99)."""
    cases = [
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, beta=0.9, n=5), (0.9510726932629295, -0.8002048397226278)),
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, beta=0.99, n=10), (1.9947359215628422, -1.0)),
        (_test_loss2, (5.0, -5.0), dict(lr=0.05, beta=0.9, n=10), (4.167485310558889, -4.149810262341654)),
    ]
    _test_method("rmsprop_metode", func, cases,
                 hint="kald jeres egen rmsprop_skridt n gange i træk, og husk at bære s videre mellem kaldene.")


def test_adam_step(func):
    """adam_skridt(a, b, va, vb, sa, sb, t, loss, lr, beta1, beta2, eps)
    -> (a, b, va, vb, sa, sb) — ét adam-skridt (momentum + rmsprop + bias-korrektion)."""
    cases = [
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, _test_loss1, 0.1, 0.9, 0.999, 1e-8),
         (0.09999999975000001, -0.0999999995, -0.3999999999999999, 0.19999999999999996,
          0.016000000000000014, 0.0040000000000000036)),
        ((5.0, -5.0, 0.0, 0.0, 0.0, 0.0, 1, _test_loss2, 0.05, 0.9, 0.999, 1e-8),
         (4.9500000000125, -4.95000000015625, 3.999999999999999, -0.31999999999999995,
          1.6000000000000014, 0.010240000000000011)),
    ]
    name = "adam_skridt"
    for inp, exp in cases:
        got = func(*inp)
        if not _eq(got, exp):
            _fail(name, f"{inp}  →  {got}  ≠  {exp}")
            print("  Hint: genbrug momentum_v_opdater (v) og rmsprop_s_opdater (s), bias-korrigér SÅ begge med t.")
            a, b, loss_fn = inp[0], inp[1], inp[7]
            if _viz: _viz.feedback_method(loss_fn, (a, b), got[:2], exp[:2])
            return
    _ok(name)


def test_adam_method(func):
    """adam_metode(loss, start, lr=0.2, beta1=0.9, beta2=0.99, eps=1e-8, n=99).
    Alle hyperparametre er eksplicitte i alle cases (ikke kun lr) — så testen forbliver
    korrekt uanset hvilke default-værdier funktionen selv sættes til."""
    cases = [
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, n=5), (0.4970442187049879, -0.49203634073565794)),
        (_test_loss1, (0.0, 0.0), dict(lr=0.1, beta1=0.9, beta2=0.99, eps=1e-8, n=10), (0.9769778845424322, -0.9267347635966232)),
        (_test_loss2, (5.0, -5.0), dict(lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, n=10), (4.502231122767305, -4.501059372865157)),
    ]
    _test_method("adam_metode", func, cases,
                 hint="kald jeres egen adam_skridt n gange i træk — t starter ved 1 og skal stige for hvert kald.")
